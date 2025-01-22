import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
import seaborn as sns
from graphdatascience import GraphDataScience
from pyspark.ml.functions import array_to_vector, vector_to_array
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from refit.v1.core.output_primary_key import _duplicate_and_null_check
from tenacity import retry, stop_after_attempt, wait_exponential

from matrix.inject import inject_object, unpack_params
from matrix.utils.pa_utils import Column, DataFrameSchema, check_output

from .encoders import AttributeEncoder
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


@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "label": Column(T.StringType(), nullable=False),
            "name": Column(T.StringType(), nullable=True),
            "property_keys": Column(T.ArrayType(T.StringType()), nullable=False),
            "property_values": Column(T.ArrayType(T.StringType()), nullable=False),
            "upstream_data_source": Column(T.ArrayType(T.StringType()), nullable=False),
        }
    )
)
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


@unpack_params()
@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "embedding": Column(T.ArrayType(T.FloatType()), nullable=False),
            "pca_embedding": Column(T.ArrayType(T.FloatType()), nullable=False),
        },
        unique=["id"],
    )
)
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


def filter_edges_for_topological_embeddings(
    nodes: ps.DataFrame, edges: ps.DataFrame, drug_types: List[str], disease_types: List[str]
) -> ps.DataFrame:
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
        return nodes.alias(column).withColumn(column, ps.functions.col("id")).select(column, "all_categories")

    df = (
        edges.alias("edges")
        .join(_create_mapping("subject"), how="left", on="subject")
        .join(_create_mapping("object"), how="left", on="object")
        # FUTURE: Improve with proper feature engineering engine
        .withColumn(
            "subject_is_drug",
            ps.functions.arrays_overlap(ps.functions.col("subject.all_categories"), ps.functions.lit(drug_types)),
        )
        .withColumn(
            "subject_is_disease",
            ps.functions.arrays_overlap(ps.functions.col("subject.all_categories"), ps.functions.lit(disease_types)),
        )
        .withColumn(
            "object_is_drug",
            ps.functions.arrays_overlap(ps.functions.col("object.all_categories"), ps.functions.lit(drug_types)),
        )
        .withColumn(
            "object_is_disease",
            ps.functions.arrays_overlap(ps.functions.col("object.all_categories"), ps.functions.lit(disease_types)),
        )
        .withColumn(
            "is_drug_disease_edge",
            (ps.functions.col("subject_is_drug") & ps.functions.col("object_is_disease"))
            | (ps.functions.col("subject_is_disease") & ps.functions.col("object_is_drug")),
        )
        .filter(~ps.functions.col("is_drug_disease_edge"))
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


def _cast_to_array(df, col: str) -> ps.DataFrame:
    if isinstance(df.schema[col].dataType, ps.types.StringType):
        return df.withColumn(
            col, ps.functions.from_json(ps.functions.col(col), ps.types.ArrayType(ps.types.FloatType()))
        )

    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "topological_embedding": Column(T.ArrayType(T.FloatType()), nullable=False),
            "pca_embedding": Column(T.ArrayType(T.FloatType()), nullable=False),
        },
        unique=["id"],
    )
)
def extract_topological_embeddings(embeddings: ps.DataFrame, nodes: ps.DataFrame, string_col: str) -> ps.DataFrame:
    """Extract topological embeddings from Neo4j and write into BQ.

    Need a conditional statement due to Node2Vec writing topological embeddings as string. Raised issue in GDS client:
    https://github.com/neo4j/graph-data-science-client/issues/742#issuecomment-2324737372.
    """
    x = (
        nodes.alias("nodes")
        .join(embeddings.transform(_cast_to_array, string_col).alias("embeddings"), on="id", how="left")
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


@has_schema(
    schema={
        "id": "string",
        "model": "string",
        "scope": "string",
        "embedding": "array<double>",
    },
    output=0,
)
# NOTE: This validates the new cache shard does _not_
# contain duplicates. This only checks a _single_ shard though,
# hence why we're also validating the result dataframe below.
# Let's avoid using the primary key statement on the cache, as that
# might be a very heavy operation.
@primary_key(primary_key=["model", "scope", "id"], output=0)
# @inline_primary_key(primary_key=["model", "scope", "id"], df="cache")
def create_node_embeddings(
    df: ps.DataFrame,
    cache: ps.DataFrame,
    batch_size: int,
    transformer,
    **transformer_kwargs,
) -> None:
    """
    Function to create node embeddings, enriching the cache and processing batches.
    Args:
        df: Input DataFrame containing nodes to process.
        cache: DataFrame holding cached embeddings.
        batch_size: Number of rows per batch.
    """
    # Load embeddings from cache
    cached_df = load_embeddings_from_cache(df, cache).cache()

    # Determine number of batches and repartition the data
    num_elements = cached_df.count()
    num_batches = (num_elements + batch_size - 1) // batch_size
    partitioned_df = cached_df.repartition(num_batches)

    # Lookup and generate missing embeddings
    enriched_df = lookup_missing_embeddings(partitioned_df, transformer, **transformer_kwargs)

    # Overwrite cache with updated embeddings
    overwrite_cache(enriched_df)


def load_embeddings_from_cache(
    dataframe: ps.DataFrame, cache: ps.DataFrame, model: str = "gpt-4", scope: str = "rtx_kg2", id_column: str = "id:ID"
) -> ps.DataFrame:
    """
    Enrich the dataframe with cached embeddings.
    Args:
        dataframe: Input DataFrame to enrich.
        cache: DataFrame containing cached embeddings.
        model: Model name used for embeddings.
        scope: Embedding enrichment scope.
        id_column: Column representing unique identifiers for input elements.
    Returns:
        DataFrame enriched with cached embeddings.
    """

    return (
        cache.filter(F.col("scope") == F.lit(scope))
        .filter(F.col("model") == F.lit(model))
        .join(dataframe, on=[F.col(id_column) == F.col("cache.id")], how="right")
    )


def lookup_missing_embeddings(df: ps.DataFrame, transformer, **transformer_kwargs) -> ps.DataFrame:
    """
    Generate embeddings for rows missing in the cache.
    Args:
        df: Input DataFrame with potential missing embeddings.
    Returns:
        DataFrame with generated embeddings for missing rows.
    """

    # Process data in parallel using mapPartitions
    return df.rdd.mapPartitions(lambda it: enrich_embeddings(it, transformer, **transformer_kwargs)).toDF()


def enrich_embeddings(iterable):
    """
    Process a batch of rows, generating embeddings for rows with null values.
    Args:
        iterable: Iterator over rows in the partition.
    Returns:
        Iterator over rows with updated embeddings.
    """
    subdf = pd.DataFrame(list(iterable))

    # Separate rows with and without embeddings
    subdf_with_embed = subdf[subdf["embedding"].notnull()]
    subdf_without_embed = subdf[subdf["embedding"].isnull()]

    # Generate embeddings for missing rows (example implementation)
    if not subdf_without_embed.empty:
        subdf_without_embed = transform(subdf_without_embed)
    # Concatenate the results and return as an iterator
    return iter(pd.concat([subdf_with_embed, subdf_without_embed], axis=0).to_dict("records"))


def overwrite_cache(df: ps.DataFrame) -> None:
    """
    Update the cache with newly generated embeddings.
    Args:
        df: DataFrame containing updated embeddings.
    """
    # Extract relevant columns for the cache
    cache_update = df.select("model", "scope", "embedding", "id").distinct()
    return cache_update


def transform(df, transformer, **transformer_kwargs):
    """Function to bucketize input data.

    Args:
        dfs: mapping of paths to df load functions
        encoder: encoder to run
    """
    return transformer.apply(df(), **transformer_kwargs)
