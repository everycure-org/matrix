import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandera import Column, DataFrameSchema, check_input
import seaborn as sns

from graphdatascience import GraphDataScience

import pyspark.sql.types as T
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from pyspark.ml.functions import array_to_vector, vector_to_array

from refit.v1.core.unpack import unpack_params

from tenacity import retry, wait_exponential, stop_after_attempt

from matrix.inject import inject_object

from .graph_algorithms import GDSGraphAlgorithm
from .encoders import AttributeEncoder

logger = logging.getLogger(__name__)


class GraphDS(GraphDataScience):
    """Adaptor class to allow injecting the GDS object.

    This is due to a drawback where refit cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth: F.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDS` instance."""
        super().__init__(endpoint, auth=tuple(auth), database=database)

        self.set_database(database)


node_schema = DataFrameSchema(
    {
        "label": Column(str),
        "id": Column(str),
        "name": Column(str),
        "property_keys": Column(object),  # array type
        "property_values": Column(object),  # array type
        "upstream_data_source": Column(object),  # array type
    },
    strict=False,
    unique=["id"],
)


@check_input(node_schema)
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


def bucketize_df(df: DataFrame, bucket_size: int, input_features: List[str], max_input_len: int):
    """Function to bucketize input dataframe.

    Function bucketizes the input dataframe in N buckets, each of size `bucket_size`
    elements. Moreover, it concatenates the `features` into a single column and limits the
    length to `max_input_len`.

    Args:
        df: Dataframe to bucketize
        attributes: to keep
        bucket_size: size of the buckets
    """

    # Order and bucketize elements
    return (
        df.transform(_bucketize, bucket_size=bucket_size)
        .withColumn(
            "text_to_embed",
            F.concat(*[F.coalesce(F.col(feature), F.lit("")) for feature in input_features]),
        )
        .withColumn("text_to_embed", F.substring(F.col("text_to_embed"), 1, max_input_len))
        .select("id", "text_to_embed", "bucket")
    )


def _bucketize(df: DataFrame, bucket_size: int) -> pd.DataFrame:
    """Function to bucketize df in given number of buckets.

    Args:
        df: dataframe to bucketize
        bucket_size: size of the buckets
    Returns:
        Dataframe augmented with `bucket` column
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

    return df.withColumn("row_num", F.row_number().over(Window.orderBy("id")) - F.lit(1)).join(
        buckets, on=[(F.col("row_num") >= (F.col("min_range"))) & (F.col("row_num") < F.col("max_range"))]
    )


@inject_object()
def compute_embeddings(
    dfs: Dict[str, Any],
    encoder: AttributeEncoder,
):
    """Function to bucketize input data.

    Args:
        dfs: mapping of paths to df load functions
        encoder: encoder to run
    """

    # NOTE: Inner function to avoid reference issues on unpacking
    # the dataframe, therefore leading to only the latest shard
    # being processed n times.
    def _func(dataframe: pd.DataFrame):
        return lambda df=dataframe: encoder.encode(df())

    shards = {}
    for path, df in dfs.items():
        # Little bit hacky, but extracting batch from hive partitioning for input path
        # As we know the input paths to this dataset are of the format /shard={num}
        bucket = path.split("/")[0].split("=")[1]

        # Invoke function to compute embeddings
        shard_path = f"bucket={bucket}/shard"
        shards[shard_path] = _func(df)

    return shards


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def compute_df_embeddings_async(df: pd.DataFrame, embedding_model) -> pd.DataFrame:
    try:
        # Embed entities in batch mode
        combined_texts = df["text_to_embed"].tolist()
        df["embedding"] = await embedding_model.aembed_documents(combined_texts)

        # Ensure floats
        df["embedding"] = df["embedding"].apply(lambda emb: np.array(emb, dtype=np.float32))
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise e

    # Drop added column
    df = df.drop(columns=["text_to_embed"])
    return df


embedding_schema = DataFrameSchema(
    {
        "embedding": Column(object, nullable=False),  # array type
        "pca_embedding": Column(object, nullable=False),  # array type
    },
    strict=True,
)


@check_input(embedding_schema)
@unpack_params()
def reduce_embeddings_dimension(df: DataFrame, transformer, input: str, output: str, skip: bool):
    return reduce_dimension(df, transformer, input, output, skip)


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
        return df.withColumn(output, F.col(input).cast("array<float>"))

    # Convert into correct type
    df = df.withColumn("features", array_to_vector(F.col(input).cast("array<float>")))

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    res = (
        transformer.fit(df)
        .transform(df)
        .withColumn(output, vector_to_array("pca_features"))
        .withColumn(output, F.col(output).cast("array<float>"))
        .drop("pca_features", "features")
    )

    return res


def filter_edges_for_topological_embeddings(
    nodes: DataFrame, edges: DataFrame, drug_types: List[str], disease_types: List[str]
):
    """Function to filter edges for topological embeddings process.

    The function removes edges connecting drug and disease nodes to avoid data leakage. Currently
    uses the `all_categories` to remove drug-disease edges.

    FUTURE: Ensure edges from ground truth dataset are explicitly removed.

    Args:
        nodes: nodes dataframe
        edges: edges dataframe
        drug_types: list of drug types
        disease_types: list of disease types
    Returns:
        Dataframe with filtered edges
    """

    def _create_mapping(column: str):
        return nodes.alias(column).withColumn(column, F.col("id")).select(column, "all_categories")

    df = (
        edges.alias("edges")
        .join(_create_mapping("subject"), how="left", on="subject")
        .join(_create_mapping("object"), how="left", on="object")
        # FUTURE: Improve with proper feature engineering engine
        .withColumn("subject_is_drug", F.arrays_overlap(F.col("subject.all_categories"), F.lit(drug_types)))
        .withColumn("subject_is_disease", F.arrays_overlap(F.col("subject.all_categories"), F.lit(disease_types)))
        .withColumn("object_is_drug", F.arrays_overlap(F.col("object.all_categories"), F.lit(drug_types)))
        .withColumn("object_is_disease", F.arrays_overlap(F.col("object.all_categories"), F.lit(disease_types)))
        .withColumn(
            "is_drug_disease_edge",
            (F.col("subject_is_drug") & F.col("object_is_disease"))
            | (F.col("subject_is_disease") & F.col("object_is_drug")),
        )
        .filter(~F.col("is_drug_disease_edge"))
        .select("edges.*")
    )

    return df


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


@unpack_params()
@inject_object()
def train_topological_embeddings(
    df: DataFrame,
    gds: GraphDataScience,
    topological_estimator: GDSGraphAlgorithm,
    projection: Any,
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

    # Validate whether the model exists
    model_name = estimator.get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    # Initialize the model
    topological_estimator.run(gds=gds, model_name=model_name, graph=graph, write_property=write_property)
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


embeddings_schema = DataFrameSchema(
    {
        "pca_embedding": Column(object, nullable=False),
        "topological_embedding": Column(object, nullable=False),
        "id": Column(object),
    },
    strict=True,
    unique=["id"],
)


@check_input(embeddings_schema)
def extract_topological_embeddings(embeddings: DataFrame, nodes: DataFrame, string_col: str) -> DataFrame:
    """Extract topological embeddings from Neo4j and write into BQ.

    Need a conditional statement due to Node2Vec writing topological embeddings as string. Raised issue in GDS client:
    https://github.com/neo4j/graph-data-science-client/issues/742#issuecomment-2324737372.
    """

    if isinstance(embeddings.schema[string_col].dataType, StringType):
        print("converting embeddings to float")
        embeddings = embeddings.withColumn(string_col, F.from_json(F.col(string_col), T.ArrayType(T.DoubleType())))

    return (
        nodes.alias("nodes")
        .join(embeddings.alias("embeddings"), on="id", how="left")
        .select("nodes.*", "embeddings.pca_embedding", "embeddings.topological_embedding")
    )


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
