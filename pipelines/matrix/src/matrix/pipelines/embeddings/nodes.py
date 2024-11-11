import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
import pandas as pd
import requests
import seaborn as sns
from graphdatascience import GraphDataScience, QueryRunner
from neo4j import Driver, GraphDatabase
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import pyspark.sql.types as T
from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from refit.v1.core.unpack import unpack_params
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .graph_algorithms import GDSGraphAlgorithm

logger = logging.getLogger(__name__)


class GraphDB:
    """Adaptor class to allow injecting the GraphDB object.

    This is due to a drawback where refit cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str | Driver | QueryRunner,
        auth: F.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDB` instance."""
        self._endpoint = endpoint
        self._auth = tuple(auth)
        self._database = database

    def driver(self):
        """Return the driver object."""
        return GraphDatabase.driver(self._endpoint, auth=self._auth)


class GraphDS(GraphDataScience):
    """Adaptor class to allow injecting the GDS object.

    This is due to a drawback where refit cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str | Driver | QueryRunner,
        auth: F.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDS` instance."""
        super().__init__(endpoint, auth=tuple(auth), database=database)

        self.set_database(database)


@has_schema(
    schema={
        "label": "string",
        "id": "string",
        "name": "string",
        "property_keys": "array<string>",
        "property_values": "array<string>",
        "upstream_data_source": "array<string>",
    },
    allow_subset=True,
)
@primary_key(primary_key=["id"])
def ingest_nodes(df: DataFrame) -> DataFrame:
    """Function to create Neo4J nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df.select("id", "name", "category", "description", "upstream_data_source")
        .withColumn("label", F.col("category"))
        # add string properties here
        .withColumn(
            "properties",
            F.create_map(
                F.lit("name"),
                F.col("name"),
                F.lit("category"),
                F.col("category"),
                F.lit("description"),
                F.col("description"),
            ),
        )
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
        # add array properties here
        .withColumn(
            "array_properties",
            F.create_map(
                F.lit("upstream_data_source"),
                F.col("upstream_data_source"),
            ),
        )
        .withColumn("array_property_keys", F.map_keys(F.col("array_properties")))
        .withColumn("array_property_values", F.map_values(F.col("array_properties")))
    )


class RateLimitException(Exception):
    """RateLimitException."""

    pass


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
)
def batch(endpoint, model, api_key, batch):
    """Function to resolve batch."""
    if len(batch) == 0:
        raise RuntimeError("Empty batch!")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"input": batch, "model": model}

    response = requests.post(f"{endpoint}/embeddings", headers=headers, json=data)

    if response.status_code == 200:
        return [item["embedding"] for item in response.json()["data"]]
    else:
        if response.status_code in [429, 500]:
            raise RateLimitException()

        print("error", response.content, response.status_code)
        raise RuntimeError()


def bucketize_nodes(
    df: DataFrame,
    attributes: str,
    bucket_size: int = 100,
):
    """Function to bucketize input dataframe.

    Args:
        df: Dataframe to bucketize
        attributes: to keep
        bucket_size: size of the buckets
    """

    # Retrieve number of elements
    num_elements = df.count()
    num_buckets = (num_elements + bucket_size - 1) // bucket_size

    # Construct df to bucketize
    spark_session: SparkSession = SparkSession.builder.getOrCreate()

    # Bucketize df
    buckets = spark_session.createDataFrame(
        data=[(bucket, bucket * bucket_size, (bucket + 1) * bucket_size) for bucket in range(num_buckets)],
        schema=["bucket", "min_range", "max_range"],
    )

    # Order and bucketize elements
    return (
        df.withColumn("row_num", F.row_number().over(Window.orderBy("id")))
        .join(buckets, on=[(F.col("row_num") >= (F.col("min_range"))) & (F.col("row_num") < F.col("max_range"))])
        .select("id", *attributes, "bucket")
    )


@inject_object()
def compute_embeddings(
    dfs: Dict[str, DataFrame],
    features: List[str],
    model: Dict[str, Any],
):
    """Function to bucketize input data.

    Args:
        df: input df
        features: features to include to compute embeddings
        config: configuration for the model
    """

    # NOTE: Inner function to avoid reference issues on unpacking
    # the dataframe, therefore leading to only the latest shard
    # being processed n times.
    def _func(dataframe: pd.DataFrame):
        return lambda: compute_df_embeddings(dataframe(), model, features)

    shards = {}
    for path, df in dfs.items():
        # Little bit hacky, but extracting batch from hive partitioning for input path
        # FUTURE: Update dataset to pass this in
        bucket = path.split("/")[0].split("=")[1]

        # Invoke function to compute embeddings
        shard_path = f"bucket={bucket}/shard"
        shards[shard_path] = _func(df)

    return shards


def compute_df_embeddings(df: pd.DataFrame, embedding_model, features: List[str]) -> pd.DataFrame:
    # Concatinate input features
    df["combined_text"] = df[features].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    # Embed entities in batch mode
    combined_texts = df["combined_text"].tolist()
    df["embedding"] = embedding_model.embed_documents(combined_texts)

    # Drop added column
    df = df.drop(columns=["combined_text"])
    return df


@unpack_params()
@inject_object()
def reduce_dimension(df: DataFrame, transformer, input: str, output: str, skip: bool):
    """Function to apply dimensionality reduction.

    Function to apply dimensionality reduction conditionally, if skip is set to true
    the original input will be returned, otherwise the given transformer will be applied.

    Args:
        df: to apply technique to
        transformer: transformer to apply
        input: name of attribute to transform
        output: name of attribute to store result
        skip: whether to skip the PCA transformation and dimensionality reduction

    Returns:
        DataFrame: A DataFrame with either the reduced dimension embeddings or the original
                   embeddings, depending on the 'skip' parameter.
    """

    breakpoint()

    if skip:
        return df.withColumn(output, F.col(input))

    # Convert into correct type
    df = df.withColumn("features", array_to_vector(input))

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    res = (
        transformer.fit(df)
        .transform(df)
        .withColumn(output, vector_to_array("pca_features"))
        .drop("pca_features", "features")
    )

    return res


def ingest_edges(nodes, edges: DataFrame):
    """Function to construct Neo4J edges."""
    return (
        edges.select(
            "subject",
            "predicate",
            "object",
            "upstream_data_source",
        )
        .withColumn("label", F.split(F.col("predicate"), ":", limit=2).getItem(1))
        # we repartition to 1 partition here to avoid deadlocks in the edges insertion of neo4j.
        # FUTURE potentially we should repartition in the future to avoid deadlocks. However
        # with edges, this is harder to do than with nodes (as they are distinct but edges have 2 nodes)
        # https://neo4j.com/docs/spark/current/performance/tuning/#parallelism
        .repartition(1)
    )


@inject_object()
def add_include_in_graphsage(df: DataFrame, gdb: GraphDB, drug_types: List[str], disease_types: List[str]) -> Dict:
    """Function to add include_in_graphsage property.

    Only edges between non drug-disease pairs are included in graphsage.
    """
    with gdb.driver() as driver:
        driver.execute_query(
            """
            MATCH (n)-[r]-(m)
            WHERE 
                n.category IN $drug_types 
                AND m.category IN $disease_types
            SET r.include_in_graphsage = 0
            """,
            database_=gdb._database,
            drug_types=drug_types,
            disease_types=disease_types,
        )

    return {"success": "true"}


@unpack_params()
@inject_object()
def train_topological_embeddings(
    df: DataFrame,
    gds: GraphDataScience,
    topological_estimator: GDSGraphAlgorithm,
    projection: Any,
    filtering: Any,
    estimator: Any,
    write_property: str,
) -> Dict:
    """Function to add graphsage embeddings.

    Function leverages the gds library to ochestrate topological embedding computation
    on the nodes of the KG.

    NOTE: The df and edges input are only added to ensure correct lineage

    Args:
        df: nodes df
        gds: the gds object
        filtering: filtering
        projection: gds projection to execute on the graph
        topological_estimator: GDS estimator to apply
        estimator: estimator to apply
        write_property: node property to write result to
    """
    # Validate whether the GDS graph exists
    graph_name = projection.get("graphName")
    if gds.graph.exists(graph_name).exists:
        graph = gds.graph.get(graph_name)
        gds.graph.drop(graph, False)
    config = projection.pop("configuration", {})
    graph, _ = gds.graph.project(*projection.values(), **config)

    # Filter out treat/GT nodes from the graph
    subgraph_name = filtering.get("graphName")
    filter_args = filtering.pop("args")
    # Drop graph if exists
    if gds.graph.exists(subgraph_name).exists:
        subgraph = gds.graph.get(subgraph_name)
        gds.graph.drop(subgraph, False)

    subgraph, _ = gds.graph.filter(subgraph_name, graph, **filter_args)

    # Validate whether the model exists
    model_name = estimator.get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    # Initialize the model
    topological_estimator.run(gds=gds, model_name=model_name, graph=subgraph, write_property=write_property)
    losses = topological_estimator.return_loss()

    # Plot convergence
    convergence = plt.figure()
    ax = convergence.add_subplot(1, 1, 1)
    ax.plot([x for x in range(len(losses))], losses)

    # Add labels and title
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Average loss per node")
    ax.set_title("Loss Chart")

    return {"success": "true"}, convergence


@inject_object()
@unpack_params()
def write_topological_embeddings(
    model: DataFrame,
    gds: GraphDataScience,
    topological_estimator: GDSGraphAlgorithm,
    projection: Any,
    estimator: Any,
    filtering: Any,
    write_property: str,
) -> Dict:
    """Write topological embeddings."""
    # Retrieve the graph
    graph_name = projection.get("graphName")
    graph = gds.graph.get(graph_name)

    # Retrieve the model
    model_name = estimator.get("modelName")
    topological_estimator.predict_write(gds=gds, model_name=model_name, graph=graph, write_property=write_property)
    return {"success": "true"}


def extract_node_embeddings(nodes: DataFrame, string_col: str) -> DataFrame:
    """Extract topological embeddings from Neo4j and write into BQ.

    Need a conditional statement due to Node2Vec writing topological embeddings as string. Raised issue in GDS client:
    https://github.com/neo4j/graph-data-science-client/issues/742#issuecomment-2324737372.
    """
    if isinstance(nodes.schema[string_col].dataType, StringType):
        # nodes = nodes.withColumn(string_col, string_to_float_list_udf(F.col(string_col)))
        nodes - nodes.withColumn(string_col, F.from_json(F.col(string_col), T.ArrayType(T.IntegerType())))
    return nodes


def visualise_pca(nodes: DataFrame, column_name: str):
    """Write topological embeddings."""
    nodes = nodes.select(column_name, "category").toPandas()
    nodes[["pca_0", "pca_1"]] = pd.DataFrame(nodes[column_name].tolist(), index=nodes.index)
    fig = plt.figure(
        figsize=(
            10,
            5,
        )
    )
    sns.scatterplot(data=nodes, x="pca_0", y="pca_1", hue="category")
    plt.suptitle("PCA scatterpot")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig


def extract_nodes_edges(nodes: DataFrame, edges: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Simple node/edge extractor function.

    Args:
        nodes: the nodes from the KG
        edges: the edges from the KG
    """
    return {"enriched_nodes": nodes, "enriched_edges": edges}
