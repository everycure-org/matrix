import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera
import pyspark.sql as ps
import seaborn as sns
from graphdatascience import GraphDataScience
from pandera.pyspark import DataFrameModel, Field
from pyspark.ml.functions import array_to_vector, vector_to_array
from tenacity import retry, stop_after_attempt, wait_exponential

from matrix.inject import inject_object, unpack_params

from .encoders import AttributeEncoder
from .features import Transform, apply_transforms
from .graph_algorithms import GDSGraphAlgorithm

logger = logging.getLogger(__name__)


class GraphDS(GraphDataScience):
    """Adaptor class to allow injecting the GDS object.

    This is due to a drawback where our functions cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth: ps.functions.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDS` instance."""
        super().__init__(endpoint, auth=tuple(auth), database=database)

        self.set_database(database)


class IngestedNodesSchema(DataFrameModel):
    id: ps.types.StringType
    label: ps.types.StringType
    name: ps.types.StringType = Field(nullable=True)
    property_keys: ps.types.ArrayType(ps.types.StringType())  # type: ignore
    property_values: ps.types.ArrayType(ps.types.StringType())  # type: ignore
    upstream_data_source: ps.types.ArrayType(ps.types.StringType())  # type: ignore


@pandera.check_output(IngestedNodesSchema)
def ingest_nodes(df: ps.DataFrame) -> ps.DataFrame:
    """Function to create Neo4J nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df.select("id", "name", "category", "description", "upstream_data_source")
        .withColumn("label", ps.functions.col("category"))
        # add string properties here
        .withColumn(
            "properties",
            ps.functions.create_map(
                ps.functions.lit("name"),
                ps.functions.col("name"),
                ps.functions.lit("category"),
                ps.functions.col("category"),
                ps.functions.lit("description"),
                ps.functions.col("description"),
            ),
        )
        .withColumn("property_keys", ps.functions.map_keys(ps.functions.col("properties")))
        .withColumn("property_values", ps.functions.map_values(ps.functions.col("properties")))
        # add array properties here
        .withColumn(
            "array_properties",
            ps.functions.create_map(
                ps.functions.lit("upstream_data_source"),
                ps.functions.col("upstream_data_source"),
            ),
        )
        .withColumn("array_property_keys", ps.functions.map_keys(ps.functions.col("array_properties")))
        .withColumn("array_property_values", ps.functions.map_values(ps.functions.col("array_properties")))
    )


def bucketize_df(df: ps.DataFrame, bucket_size: int, input_features: List[str], max_input_len: int) -> ps.DataFrame:
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
            ps.functions.concat(
                *[ps.functions.coalesce(ps.functions.col(feature), ps.functions.lit("")) for feature in input_features]
            ),
        )
        .withColumn("text_to_embed", ps.functions.substring(ps.functions.col("text_to_embed"), 1, max_input_len))
        .select("id", "text_to_embed", "bucket")
    )


def _bucketize(df: ps.DataFrame, bucket_size: int) -> ps.DataFrame:
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
    spark_session: ps.SparkSession = ps.SparkSession.builder.getOrCreate()

    # Bucketize df
    buckets = spark_session.createDataFrame(
        data=[(bucket, bucket * bucket_size, (bucket + 1) * bucket_size) for bucket in range(num_buckets)],
        schema=["bucket", "min_range", "max_range"],
    )

    return df.withColumn(
        "row_num", ps.functions.row_number().over(ps.window.Window.orderBy("id")) - ps.functions.lit(1)
    ).join(
        buckets,
        on=[
            (ps.functions.col("row_num") >= (ps.functions.col("min_range")))
            & (ps.functions.col("row_num") < ps.functions.col("max_range"))
        ],
    )


@inject_object()
def compute_embeddings(
    dfs: Dict[str, Any],
    encoder: AttributeEncoder,
) -> Dict[str, Any]:
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


class EmbeddingSchema(DataFrameModel):
    id: ps.types.StringType
    embedding: ps.types.ArrayType(ps.types.FloatType(), True)  # type: ignore
    pca_embedding: ps.types.ArrayType(ps.types.FloatType(), True)  # type: ignore

    class Config:
        strict = False
        unique = ["id"]


@pandera.check_output(EmbeddingSchema)
@unpack_params()
def reduce_embeddings_dimension(df: ps.DataFrame, transformer, input: str, output: str, skip: bool) -> ps.DataFrame:
    return reduce_dimension(df, transformer, input, output, skip)


@unpack_params()
@inject_object()
def reduce_dimension(df: ps.DataFrame, transformer, input: str, output: str, skip: bool) -> ps.DataFrame:
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
        return df.withColumn(output, ps.functions.col(input).cast("array<float>"))

    # Convert into correct type
    df = df.withColumn("features", array_to_vector(ps.functions.col(input).cast("array<float>")))

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    res = (
        transformer.fit(df)
        .transform(df)
        .withColumn(output, vector_to_array("pca_features"))
        .withColumn(output, ps.functions.col(output).cast("array<float>"))
        .withColumn("pca_embedding", ps.functions.col(output))
        .drop("pca_features", "features")
    )

    return res


@inject_object()
def filter_edges_for_topological_embeddings(
    nodes: ps.DataFrame,
    edges: ps.DataFrame,
    transformations: List[Transform],
):
    """Function to filter edges for topological embeddings process.

    The function removes edges connecting drug and disease nodes to avoid data leakage. Currently
    uses the `all_categories` to remove drug-disease edges.

    FUTURE: Ensure edges from ground truth dataset are explicitly removed.

    Args:
        nodes: nodes dataframe
        edges: edges dataframe
        transformations: to apply
    Returns:
        Dataframe with filtered edges
    """

    def _create_mapping(column: str):
        return nodes.alias(column).withColumn(column, ps.functions.col("id")).select(column, "all_categories")

    df = (
        edges.alias("edges")
        .join(_create_mapping("subject"), how="left", on="subject")
        .join(_create_mapping("object"), how="left", on="object")
        .transform(apply_transforms, transformations)
        .select("edges.*")
    )

    return df


def ingest_edges(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    """Function to construct Neo4J edges."""
    return (
        edges.select(
            "subject",
            "predicate",
            "object",
            "upstream_data_source",
        )
        .withColumn("label", ps.functions.split(ps.functions.col("predicate"), ":", limit=2).getItem(1))
        # we repartition to 1 partition here to avoid deadlocks in the edges insertion of neo4j.
        # FUTURE potentially we should repartition in the future to avoid deadlocks. However
        # with edges, this is harder to do than with nodes (as they are distinct but edges have 2 nodes)
        # https://neo4j.com/docs/spark/current/performance/tuning/#parallelism
        .repartition(1)
    )


@unpack_params()
@inject_object()
def train_topological_embeddings(
    df: ps.DataFrame,
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
    model: ps.DataFrame,
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


class ExtractedTopologicalEmbeddingSchema(DataFrameModel):
    id: ps.types.StringType
    topological_embedding: ps.types.ArrayType(ps.types.FloatType(), True) = Field(nullable=True)  # type: ignore
    pca_embedding: ps.types.ArrayType(ps.types.FloatType(), True) = Field(nullable=True)  # type: ignore

    class Config:
        strict = False
        unique = ["id"]


@pandera.check_output(ExtractedTopologicalEmbeddingSchema)
def extract_topological_embeddings(embeddings: ps.DataFrame, nodes: ps.DataFrame, string_col: str) -> ps.DataFrame:
    """Extract topological embeddings from Neo4j and write into BQ.

    Need a conditional statement due to Node2Vec writing topological embeddings as string. Raised issue in GDS client:
    https://github.com/neo4j/graph-data-science-client/issues/742#issuecomment-2324737372.
    """

    if isinstance(embeddings.schema[string_col].dataType, ps.types.StringType):
        print("converting embeddings to float")
        embeddings = embeddings.withColumn(
            string_col, ps.functions.from_json(ps.functions.col(string_col), ps.types.ArrayType(ps.types.FloatType()))
        )

    x = (
        nodes.alias("nodes")
        .join(embeddings.alias("embeddings"), on="id", how="left")
        .select("nodes.*", "embeddings.pca_embedding", "embeddings.topological_embedding")
        .withColumn("pca_embedding", ps.functions.col("pca_embedding").cast("array<float>"))
        .withColumn("topological_embedding", ps.functions.col("topological_embedding").cast("array<float>"))
    )
    return x


def visualise_pca(nodes: ps.DataFrame, column_name: str) -> plt.Figure:
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
