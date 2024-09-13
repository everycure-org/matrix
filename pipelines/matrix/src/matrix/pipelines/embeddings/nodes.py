"""Nodes for embeddings pipeline."""

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from graphdatascience import GraphDataScience, QueryRunner
from matplotlib.pyplot import plot
from neo4j import Driver, GraphDatabase
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.window import Window
from refit.v1.core.inject import inject_object
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from refit.v1.core.unpack import unpack_params
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .graph_algorithms import *

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
        "kg_sources": "array<string>",
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
        df.select("id", "name", "category", "description", "kg_sources")
        .withColumn("label", F.split(F.col("category"), ":", limit=2).getItem(1))
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
    )


class RateLimitException(Exception):
    """RateLimitException."""

    pass


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(RateLimitException),
)
def batch(endpoint, model, api_key, batch):
    """Function to resolve batch."""
    print(f"processing batch with length {len(batch)}")

    if len(batch) == 0:
        raise RuntimeError("Empty batch!")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"input": batch, "model": model}

    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 200:
        return [item["embedding"] for item in response.json()["data"]]
    else:
        if response.status_code in [429, 500]:
            print(f"rate limit")
            raise RateLimitException()

        print("error", response.content, response.status_code)
        raise RuntimeError()


@unpack_params()
@inject_object()
def compute_embeddings(
    input: DataFrame,
    features: List[str],
    attribute: str,
    api_key: str,
    batch_size: int,
    endpoint: str,
    model: str,
    concurrency: int,
):
    """Function to orchestrate embedding computation in Neo4j.

    Args:
        input: input df
        gdb: graph database instance
        features: features to include to compute embeddings
        api_key: api key to use
        batch_size: batch size
        attribute: attribute to add
        endpoint: endpoint to use
        model: model to use
        concurrency: number of concurrent calls to execute
    """
    batch_udf = F.udf(
        lambda z: batch(endpoint, model, api_key, z), ArrayType(ArrayType(FloatType()))
    )

    window = Window.orderBy(F.lit(1))

    res = (
        input.withColumn("row_num", F.row_number().over(window))
        .repartition(128)
        .withColumn("batch", F.floor((F.col("row_num") - 1) / batch_size))
        # NOTE: There is quite a lot of nodes without name and description, thereby resulting
        # in embeddings of the empty string.
        .withColumn(
            "input",
            F.concat(*[F.coalesce(F.col(feature), F.lit("")) for feature in features]),
        )
        .withColumn("input", F.substring(F.col("input"), 1, 512))  # TODO: Extract param
        .groupBy("batch")
        .agg(
            F.collect_list("id").alias("id"),
            F.collect_list("input").alias("input"),
        )
        .repartition(128)
        # .withColumn("num_ids", F.size(F.col("id")))
        # .withColumn("num_input", F.size(F.col("input")))
        # .withColumn("validated",  F.when(F.col("num_ids") == F.col("num_input"), True).otherwise(False))
        .withColumn(attribute, batch_udf(F.col("input")))
        .withColumn("_conc", F.arrays_zip(F.col("id"), F.col(attribute)))
        .withColumn("exploded", F.explode(F.col("_conc")))
        .select(
            F.col("exploded.id").alias("id"),
            F.col(f"exploded.{attribute}").alias(attribute),
        )
        .repartition(128)
        .join(input, on="id")
    )

    return res


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
            "subject", "predicate", "object", "knowledge_sources", "kg_sources"
        )
        .withColumn("label", F.split(F.col("predicate"), ":", limit=2).getItem(1))
        # we repartition to 1 partition here to avoid deadlocks in the edges insertion of neo4j.
        # FUTURE potentially we should repartition in the future to avoid deadlocks. However
        # with edges, this is harder to do than with nodes (as they are distinct but edges have 2 nodes)
        # https://neo4j.com/docs/spark/current/performance/tuning/#parallelism
        .repartition(1)
    )


@inject_object()
def add_include_in_graphsage(
    df: DataFrame, gdb: GraphDB, drug_types: List[str], disease_types: List[str]
) -> Dict:
    """Function to add include_in_graphsage property.

    Only edges between non drug-disease pairs are included in graphsage.
    """
    with gdb.driver() as driver:
        q = driver.execute_query(
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
    topological_estimator.run(
        gds=gds, model_name=model_name, graph=subgraph, write_property=write_property
    )
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
    topological_estimator.predict_write(
        gds=gds, model_name=model_name, graph=graph, write_property=write_property
    )
    return {"success": "true"}


def string_to_float_list(s: str) -> List[float]:
    """UDF to transform str into list. Fix for Node2Vec array being written as string."""
    if s is not None:
        return [float(x) for x in s.strip()[1:-1].split(",")]
    return []


def extract_node_embeddings(nodes: DataFrame, string_col: str) -> DataFrame:
    """Extract topological embeddings from Neo4j and write into BQ.

    Need a conditional statement due to Node2Vec writing topological embeddings as string. Raised issue in GDS client:
    https://github.com/neo4j/graph-data-science-client/issues/742#issuecomment-2324737372.
    """
    if isinstance(nodes.schema[string_col].dataType, StringType):
        string_to_float_list_udf = udf(string_to_float_list, ArrayType(FloatType()))
        nodes = nodes.withColumn(
            string_col, string_to_float_list_udf(F.col(string_col))
        )
    return nodes


def visualise_pca(nodes: DataFrame, column_name: str):
    """Write topological embeddings."""
    nodes = nodes.select(column_name, "category").toPandas()
    nodes[["pca_0", "pca_1"]] = pd.DataFrame(
        nodes[column_name].tolist(), index=nodes.index
    )
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
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize="small"
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig


def extract_nodes_edges(
    nodes: DataFrame, edges: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Simple node/edge extractor function.

    Args:
        nodes: the nodes from the KG
        edges: the edges from the KG
    """
    return {"enriched_nodes": nodes, "enriched_edges": edges}
