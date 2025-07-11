import logging
from functools import partial, reduce

import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
from joblib import Memory
from matrix_schema.datamodel.pandera import get_matrix_edge_schema, get_matrix_node_schema
from matrix_schema.utils.pandera_utils import Column, DataFrameSchema, check_output
from pyspark.sql.window import Window

from matrix.inject import inject_object
from matrix.pipelines.integration.filters import determine_most_specific_category

# TODO move these into config
memory = Memory(location=".cache/nodenorm", verbose=0)
logger = logging.getLogger(__name__)


@inject_object()
@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
        },
        unique=["id"],
    ),
    df_name="nodes",
)
@check_output(
    DataFrameSchema(
        columns={
            "subject": Column(T.StringType(), nullable=False),
            "predicate": Column(T.StringType(), nullable=False),
            "object": Column(T.StringType(), nullable=False),
        },
    ),
    df_name="edges",
    raise_df_undefined=False,
)
def transform(transformer, **kwargs) -> dict[str, ps.DataFrame]:
    return transformer.transform(**kwargs)


@check_output(
    schema=get_matrix_edge_schema(validate_enumeration_values=False),
    pass_columns=True,
)
def union_edges(*edges, cols: list[str]) -> ps.DataFrame:
    """Function to unify edges datasets."""
    unioned_dataset = (
        _union_datasets(*edges)
        .groupBy(["subject", "predicate", "object"])
        .agg(
            F.flatten(F.collect_set("upstream_data_source")).alias("upstream_data_source"),
            # TODO: we shouldn't just take the first one but collect these values from multiple upstream sources
            F.first("knowledge_level", ignorenulls=True).alias("knowledge_level"),
            F.first("agent_type", ignorenulls=True).alias("agent_type"),
            F.first("subject_aspect_qualifier", ignorenulls=True).alias("subject_aspect_qualifier"),
            F.first("subject_direction_qualifier", ignorenulls=True).alias("subject_direction_qualifier"),
            F.first("object_direction_qualifier", ignorenulls=True).alias("object_direction_qualifier"),
            F.first("object_aspect_qualifier", ignorenulls=True).alias("object_aspect_qualifier"),
            F.first("primary_knowledge_source", ignorenulls=True).alias("primary_knowledge_source"),
            F.flatten(F.collect_set("aggregator_knowledge_source")).alias("aggregator_knowledge_source"),
            F.flatten(F.collect_set("publications")).alias("publications"),
            F.max("num_references").cast(T.IntegerType()).alias("num_references"),
            F.max("num_sentences").cast(T.IntegerType()).alias("num_sentences"),
        )
        .select(*cols)
    )
    return unioned_dataset


@check_output(
    schema=get_matrix_node_schema(validate_enumeration_values=False),
    pass_columns=True,
)
def union_and_deduplicate_nodes(retrieve_most_specific_category: bool, *nodes, cols: list[str]) -> ps.DataFrame:
    """Function to unify nodes datasets."""
    unioned_datasets = (
        _union_datasets(*nodes)
        # first we group the dataset by id to deduplicate
        .groupBy("id")
        .agg(
            F.first("name", ignorenulls=True).alias("name"),
            F.first("category", ignorenulls=True).alias("category"),
            F.first("description", ignorenulls=True).alias("description"),
            F.first("international_resource_identifier", ignorenulls=True).alias("international_resource_identifier"),
            F.flatten(F.collect_set("equivalent_identifiers")).alias("equivalent_identifiers"),
            F.flatten(F.collect_set("all_categories")).alias("all_categories"),
            F.flatten(F.collect_set("labels")).alias("labels"),
            F.flatten(F.collect_set("publications")).alias("publications"),
            F.flatten(F.collect_set("upstream_data_source")).alias("upstream_data_source"),
        )
    )
    # next we need to apply a number of transformations to the nodes to ensure grouping by id did not select wrong information
    # this is especially important if we integrate multiple KGs

    if retrieve_most_specific_category:
        unioned_datasets = unioned_datasets.transform(determine_most_specific_category)

    return unioned_datasets.select(*cols)


def _union_datasets(
    *datasets: ps.DataFrame,
) -> ps.DataFrame:
    """
    Helper function to unify datasets and deduplicate them.
    Args:
        datasets: List of dataset names to unify.

    Returns:
        A unified and deduplicated DataFrame.
    """
    return reduce(partial(ps.DataFrame.unionByName, allowMissingColumns=True), datasets)


@check_output(
    DataFrameSchema(
        columns={
            "normalization_success": Column(T.BooleanType(), nullable=False),
        },
    ),
)
def _format_mapping_df(mapping_df: ps.DataFrame) -> ps.DataFrame:
    return (
        mapping_df.select("id", "normalized_id")
        .withColumn(
            "normalization_success",
            F.when((F.col("normalized_id").isNotNull() | (F.col("normalized_id") != "None")), True).otherwise(False),
        )
        # avoids nulls in id column, if we couldn't resolve IDs, we keep original
        .withColumn("normalized_id", F.coalesce(F.col("normalized_id"), F.col("id")))
    )


def normalize_edges(
    mapping_df: ps.DataFrame,
    edges: ps.DataFrame,
) -> ps.DataFrame:
    """Function normalizes a KG using external API endpoint.

    This function takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with normalized IDs.
    """
    mapping_df = _format_mapping_df(mapping_df)
    # edges are a bit more complex, we need to map both the subject and object
    subject_normalized_mapping_df = mapping_df.withColumnsRenamed(
        {
            "id": "subject",
            "normalized_id": "subject_normalized",
            "normalization_success": "subject_normalization_success",
        }
    )
    edges = edges.join(subject_normalized_mapping_df, on="subject", how="left")
    edges = edges.withColumn("subject_normalized", F.coalesce("subject_normalized", "subject"))

    object_normalized_mapping_df = mapping_df.withColumnsRenamed(
        {
            "id": "object",
            "normalized_id": "object_normalized",
            "normalization_success": "object_normalization_success",
        }
    )
    edges = edges.join(object_normalized_mapping_df, on="object", how="left")
    edges = edges.withColumn("object_normalized", F.coalesce("object_normalized", "object"))

    edges = edges.withColumnsRenamed({"subject": "original_subject", "object": "original_object"})
    edges = edges.withColumnsRenamed({"subject_normalized": "subject", "object_normalized": "object"})

    edges = edges.dropDuplicates(subset=["subject", "predicate", "object"])

    return edges


def normalize_nodes(
    mapping_df: ps.DataFrame,
    nodes: ps.DataFrame,
) -> ps.DataFrame:
    """Function normalizes a KG using external API endpoint.

    This function takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with normalized IDs.

    """
    mapping_df = _format_mapping_df(mapping_df)

    # add normalized_id to nodes
    return (
        nodes.join(mapping_df, on="id", how="left")
        .withColumnsRenamed({"id": "original_id"})
        .withColumnsRenamed({"normalized_id": "id"})
        # Ensure deduplicated
        .withColumn("_rn", F.row_number().over(Window.partitionBy("id").orderBy(F.col("original_id"))))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def check_nodes_and_edges_matching(edges: ps.DataFrame, nodes: ps.DataFrame):
    """
    Function examining if all nodes and edges are matching post-normalization.

    All subjects/objects within edges dataframe should be found in nodes dataframe. If there are subject/object
    identifiers which are not matching with nodes dataframe, error will be thrown.

    Parameters
    ----------
    edges : pyspark.sql.DataFrame
        A DataFrame containing normalized edges, including columns such as
        `subject`, `object`, `original_subject`, `original_object`,
        `subject_normalization_success`, and `object_normalization_success`.
    nodes : pyspark.sql.DataFrame
        A DataFrame containing normalized nodes, including `id` and optionally `category`.

    Returns
    -------
    Nothing returned; error is thrown if mismatching
    """
    match_count = (
        edges.join(nodes.withColumnRenamed("id", "subject"), on="subject", how="inner")
        .join(nodes.withColumnRenamed("id", "object"), on="object", how="inner")
        .count()
    )
    if edges.count() != match_count:
        raise Exception("Nodes and Edges are mismatching post-normalization; please investigate")
    return None


@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "original_id": Column(T.StringType(), nullable=False),
            "category": Column(T.StringType(), nullable=True),
            "normalization_success": Column(T.BooleanType(), nullable=True),
            "upstream_data_source": Column(T.StringType(), nullable=False),
            "source_role": Column(T.StringType(), nullable=False),
        }
    )
)
def normalization_summary_nodes_and_edges(
    edges: ps.DataFrame,
    nodes: ps.DataFrame,
    source: str,
) -> ps.DataFrame:
    """
    Generate a flattened per-source summary of node normalization success for both subjects and objects in edges.

    This function processes a set of normalized edges and nodes to extract normalization outcomes,
    linking original IDs with their normalized equivalents, and annotating each entry with source roles
    ("subject" or "object"), categories (if available), and the originating data source.

    Parameters
    ----------
    edges : pyspark.sql.DataFrame
        A DataFrame containing normalized edges, including columns such as
        `subject`, `object`, `original_subject`, `original_object`,
        `subject_normalization_success`, and `object_normalization_success`.
    nodes : pyspark.sql.DataFrame
        A DataFrame containing normalized nodes, including `id` and optionally `category`.
    source : str
        The name of the upstream data source providing the edges and nodes.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame summarizing normalization outcomes for all nodes referenced in the edge file.
        Columns include:
        - `id`: Normalized CURIE of the node
        - `original_id`: Original (pre-normalization) CURIE
        - `normalization_success`: Boolean or flag indicating whether normalization was successful
        - `category`: Biolink category assigned to the node (if available)
        - `source_role`: Indicates whether the node was a subject or object in the edge
        - `upstream_data_source`: Name of the originating data source
    """

    # Check nodes and edges matching post-normalization
    check_nodes_and_edges_matching(edges, nodes)

    # Safe fallback for category column
    if "category" in nodes.columns:
        nodes_for_join = nodes.select("id", "category")
    else:
        nodes_for_join = nodes.select("id").withColumn("category", F.lit(None).cast("string"))

    return (
        edges.selectExpr(
            "subject as id", "original_subject as original_id", "subject_normalization_success as normalization_success"
        )
        .join(nodes_for_join, on="id", how="left")
        .withColumn("source_role", F.lit("subject"))
        .unionByName(
            edges.selectExpr(
                "object as id",
                "original_object as original_id",
                "object_normalization_success as normalization_success",
            )
            .join(nodes_for_join, on="id", how="left")
            .withColumn("source_role", F.lit("object"))
        )
        .withColumn("upstream_data_source", F.lit(source))
    )


def normalization_summary_nodes_only(
    nodes: ps.DataFrame,
    source: str,
) -> ps.DataFrame:
    """
    Generate a flattened per-source summary of node normalization success from a node-only dataset.

    This function processes a set of normalized nodes and returns a summary table that includes
    the normalized ID, original ID, category (if available), and normalization success flag,
    along with source metadata for tracking.

    Parameters
    ----------
    nodes : pyspark.sql.DataFrame
        A DataFrame containing normalized nodes with at least `id`, `original_id`,
        and `normalization_success` columns. The `category` column is optional.
    source : str
        The name of the upstream data source providing the nodes.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame summarizing normalization outcomes for all nodes.
        Columns include:
        - `id`: Normalized CURIE of the node
        - `original_id`: Original (pre-normalization) CURIE
        - `category`: Biolink category assigned to the node (if available)
        - `normalization_success`: Boolean or flag indicating whether normalization was successful
        - `source_role`: Fixed value "node" indicating the entity type
        - `upstream_data_source`: Name of the originating data source
    """

    if "category" in nodes.columns:
        selected = nodes.select("id", "original_id", "category", "normalization_success")
    else:
        selected = nodes.select("id", "original_id", "normalization_success").withColumn(
            "category", F.lit(None).cast("string")
        )

    return selected.withColumn("source_role", F.lit("node")).withColumn("upstream_data_source", F.lit(source))
