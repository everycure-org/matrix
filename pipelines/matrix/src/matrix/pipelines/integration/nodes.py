import logging
from functools import partial, reduce

import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
from joblib import Memory
from matrix_schema.datamodel.pandera import get_matrix_edge_schema, get_matrix_node_schema, get_unioned_edge_schema
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
    schema=get_unioned_edge_schema(validate_enumeration_values=False),
    pass_columns=True,
)
def union_edges(core_id_mapping: ps.DataFrame, *edges, cols: list[str]) -> ps.DataFrame:
    """Function to unify edges datasets and promote subject/object to core_id."""

    unioned_edges = _union_datasets(*edges)

    # Promote subject to core_id
    unioned_edges = (
        unioned_edges.join(core_id_mapping.withColumnRenamed("normalized_id", "subject"), on="subject", how="left")
        .withColumn("subject", F.coalesce("core_id", "subject"))
        .drop("core_id")
    )

    # Promote object to core_id
    unioned_edges = (
        unioned_edges.join(core_id_mapping.withColumnRenamed("normalized_id", "object"), on="object", how="left")
        .withColumn("object", F.coalesce("core_id", "object"))
        .drop("core_id")
    )

    unioned_dataset = (
        unioned_edges.groupBy(["subject", "predicate", "object"])
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
            F.collect_set(
                F.when(F.col("primary_knowledge_source").isNotNull(), F.col("primary_knowledge_source"))
            ).alias("primary_knowledge_sources"),
            F.flatten(F.collect_set("publications")).alias("publications"),
            F.max("num_references").cast(T.IntegerType()).alias("num_references"),
            F.max("num_sentences").cast(T.IntegerType()).alias("num_sentences"),
        )
        .select(*cols)
    )
    return unioned_dataset


@check_output(
    schema=get_matrix_edge_schema(validate_enumeration_values=False),
    pass_columns=True,
)
def unify_ground_truth(*edges) -> ps.DataFrame:
    """Function to unify edges datasets."""
    # fmt: off
    return _union_datasets(*edges)
    # fmt: on


@check_output(
    schema=get_matrix_node_schema(validate_enumeration_values=False),
    pass_columns=True,
)
def union_and_deduplicate_nodes(
    retrieve_most_specific_category: bool, core_id_mapping: ps.DataFrame, *nodes, cols: list[str]
) -> ps.DataFrame:
    """Function to unify nodes datasets."""

    unioned_nodes = _union_datasets(*nodes)

    # Promote core nodes to use their core_id as the canonical id and core_name as canonical name
    # non-core nodes keep their normalized id and original name
    core_promoted_nodes = (
        unioned_nodes.join(core_id_mapping.withColumnRenamed("normalized_id", "id"), on="id", how="left")
        .withColumn("id", F.coalesce("core_id", "id"))
        .withColumn("name", F.coalesce("core_name", "name"))
        .drop("core_id", "core_name")
    )

    unioned_datasets = (
        core_promoted_nodes
        # first we group the dataset by id to deduplicate
        .groupBy("id").agg(
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
        mapping_df.withColumn("normalized_id", F.col("normalization_struct.normalized_id"))
        .withColumn("normalized_categories", F.col("normalization_struct.normalized_categories"))
        .drop("normalization_struct")
        .select("id", "normalized_id", "normalized_categories")
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
    mapping_df = _format_mapping_df(mapping_df).select("id", "normalized_id", "normalization_success")

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


def _normalize_nodes_base(
    mapping_df: ps.DataFrame,
    nodes: ps.DataFrame,
) -> ps.DataFrame:
    """Base normalization function.

    This function handles the core normalization logic but leaves
    deduplication to the calling functions so they can control
    which rows to keep when there are conflicts.
    """
    mapping_df = _format_mapping_df(mapping_df)

    nodes_normalized = (
        nodes.join(
            mapping_df.select("id", "normalized_id", "normalized_categories", "normalization_success"),
            on="id",
            how="left",
        )
        .withColumnRenamed("id", "original_id")
        .withColumnRenamed("normalized_id", "id")
    )

    # Determine the value of `all_categories` based on normalization results:
    # - If `normalized_categories` is non-null and non-empty, use it.
    # - Otherwise, if `original_categories` exists, fall back to it.
    # - If neither is available, use an empty list as the default.
    if "all_categories" in nodes_normalized.columns:
        nodes_normalized = nodes_normalized.withColumnRenamed("all_categories", "original_categories")
    else:
        nodes_normalized = nodes_normalized.withColumn(
            "original_categories", F.lit([]).cast(T.ArrayType(T.StringType()))
        )

    # Determine the value of `all_categories` based on normalization results:
    # - If `normalized_categories` is non-null and non-empty, use it.
    # - Otherwise, if `original_categories` exists, fall back to it.
    # - If neither is available, use an empty list as the default.
    nodes_normalized = nodes_normalized.withColumn(
        "all_categories",
        F.when(
            (F.col("normalized_categories").isNotNull()) & (F.size(F.col("normalized_categories")) > 0),
            F.col("normalized_categories"),
        )
        .when(F.col("original_categories").isNotNull(), F.col("original_categories"))
        .otherwise(F.lit([]).cast(T.ArrayType(T.StringType()))),
    )

    return nodes_normalized


def normalize_nodes(
    mapping_df: ps.DataFrame,
    nodes: ps.DataFrame,
) -> ps.DataFrame:
    """Function normalizes a KG using external API endpoint.

    This function takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with normalized IDs.

    """
    nodes_normalized = _normalize_nodes_base(mapping_df, nodes)

    # Deduplicate rows by id
    return (
        nodes_normalized.withColumn("_rn", F.row_number().over(Window.partitionBy("id").orderBy("original_id")))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def normalize_core_nodes(
    mapping_df: ps.DataFrame,
    nodes: ps.DataFrame,
) -> ps.DataFrame:
    """Function normalizes core nodes (drugs/diseases) using external API endpoint.

    This function takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with normalized IDs.

    """
    nodes_normalized = _normalize_nodes_base(mapping_df, nodes)

    # If this is a core source (drug and disease list), add core_id = original_id
    # The core_id will be used as the id for all equivalent nodes  in the merged graph
    nodes_normalized = nodes_normalized.withColumn("core_id", F.col("original_id"))

    # Intra-source conflict detection: same normalized ID mapping to multiple core_ids
    # checking here ensures that there are no conflicts at the per-source level
    conflicts = (
        nodes_normalized.groupBy("id")
        .agg(F.countDistinct("core_id").alias("distinct_core_ids"))
        .filter(F.col("distinct_core_ids") > 1)
    )

    conflict_count = conflicts.count()
    if conflict_count > 0:
        logger.error(
            f"{conflict_count} normalized IDs map to multiple core_ids. "
            f"Multiple core_ids found for the same normalized_id. "
            f"Please fix source data."
        )
        conflicts.show(truncate=False)
        raise Exception("Normalized ID conflicts detected; please investigate")
    else:
        logger.info("No normalized ID conflicts found.")

    # Deduplicate rows by id
    return (
        nodes_normalized.withColumn("_rn", F.row_number().over(Window.partitionBy("id").orderBy("original_id")))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def create_core_id_mapping(*nodes: ps.DataFrame) -> ps.DataFrame:
    """Creates a mapping from normalized_id to core_id for core sources."""
    df = _union_datasets(*nodes)

    df_filtered = df.select("id", "core_id", "name").filter(  # 'id' is already the normalized_id at this point
        (F.col("id").isNotNull()) & (F.col("core_id").isNotNull())
    )

    # Inter-source conflict detection: same normalized ID mapping to multiple core_ids
    # Checking at this step ensures that there are no duplicates across all core sources
    conflicts = (
        df_filtered.groupBy("id")
        .agg(F.countDistinct("core_id").alias("distinct_core_ids"))
        .filter(F.col("distinct_core_ids") > 1)
    )

    conflict_count = conflicts.count()
    if conflict_count > 0:
        logger.warning(
            f"{conflict_count} normalized IDs map to multiple core_ids. "
            f"Proceeding despite core_id conflicts. Multiple core_ids found for the same normalized_id. "
            f"Recommened to fix source data."
        )
        conflicts.show(50, truncate=False)

    return (
        df_filtered.dropDuplicates(["id"])
        .withColumnRenamed("id", "normalized_id")  # Ensure consistent naming for join
        .withColumnRenamed("name", "core_name")
        .select("normalized_id", "core_id", "core_name")
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
    edge_ids = edges.select(F.col("subject").alias("id")).union(edges.select(F.col("object").alias("id"))).distinct()
    node_ids = nodes.select("id").distinct()

    missing_ids = edge_ids.join(node_ids, on="id", how="left_anti")

    missing_count = missing_ids.count()
    if missing_count > 0:
        logger.warning(f"{missing_count} edges refer to node IDs not found in nodes.")
        missing_ids.show(50, truncate=False)
        raise Exception("Nodes and Edges are mismatching; please investigate")
    else:
        logger.info("All edge node references are valid.")
        return "validation_passed"


@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "original_id": Column(T.StringType(), nullable=False),
            "normalization_success": Column(T.BooleanType(), nullable=True),
            "original_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "normalized_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "all_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "source_role": Column(T.StringType(), nullable=False),
            "upstream_data_source": Column(T.StringType(), nullable=False),
        }
    )
)
def normalization_summary_nodes_and_edges(
    edges: ps.DataFrame,
    nodes: ps.DataFrame,
    mapping_df: ps.DataFrame,
    source: str,
) -> ps.DataFrame:
    """
    Summarize normalization outcomes for all nodes referenced in an edge set.

    Args:
        edges (pyspark.sql.DataFrame):
            Normalized edge data including subject/object and original IDs.
        nodes (pyspark.sql.DataFrame):
            Normalized node data, including original and final category assignments.
        mapping_df (pyspark.sql.DataFrame):
            Mapping output containing normalized_categories per node ID.
        source (str):
            Name of the upstream data source.

    Returns:
        pyspark.sql.DataFrame:
            Flattened summary DataFrame with columns:
                - id: Normalized node ID
                - original_id: Original node ID
                - normalization_success: Whether normalization succeeded
                - original_categories: Pre-normalization categories
                - normalized_categories: Categories returned by the normalizer
                - all_categories: Final categories used
                - source_role: 'subject' or 'object'
                - upstream_data_source: Name of the data source
    """

    formatted_mapping = mapping_df.select(
        "id",
        F.col("normalization_struct.normalized_categories")
        .cast(T.ArrayType(T.StringType()))
        .alias("normalized_categories"),
    )

    # Check nodes and edges matching post-normalization
    check_nodes_and_edges_matching(edges, nodes)

    nodes_for_join = nodes.select("id", "original_categories", "all_categories").join(
        formatted_mapping, on="id", how="left"
    )

    def summarize_role(role_edges: ps.DataFrame, role_nodes_to_join: ps.DataFrame, role: str):
        return (
            role_edges.selectExpr(
                f"{role} as id",
                f"original_{role} as original_id",
                f"{role}_normalization_success as normalization_success",
            )
            .join(role_nodes_to_join, on="id", how="left")
            .withColumn("source_role", F.lit(role))
        )

    return (
        summarize_role(edges, nodes_for_join, "subject")
        .unionByName(summarize_role(edges, nodes_for_join, "object"))
        .withColumn("upstream_data_source", F.lit(source))
        .select(
            "id",
            "original_id",
            "normalization_success",
            "original_categories",
            "normalized_categories",
            "all_categories",
            "source_role",
            "upstream_data_source",
        )
    )


@check_output(
    DataFrameSchema(
        columns={
            "id": Column(T.StringType(), nullable=False),
            "original_id": Column(T.StringType(), nullable=False),
            "normalization_success": Column(T.BooleanType(), nullable=True),
            "original_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "normalized_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "all_categories": Column(T.ArrayType(T.StringType()), nullable=True),
            "source_role": Column(T.StringType(), nullable=False),
            "upstream_data_source": Column(T.StringType(), nullable=False),
        }
    )
)
def normalization_summary_nodes_only(
    nodes: ps.DataFrame,
    mapping_df: ps.DataFrame,
    source: str,
) -> ps.DataFrame:
    """
    Summarize normalization outcomes for a node-only dataset.

    Args:
        nodes (pyspark.sql.DataFrame):
            Normalized nodes, including original and final category assignments.
        mapping_df (pyspark.sql.DataFrame):
            Mapping output containing normalized_categories per node ID.
        source (str):
            Name of the upstream data source.

    Returns:
        pyspark.sql.DataFrame:
            Summary DataFrame with the following columns:
                - id: Normalized node ID
                - original_id: Original node ID
                ...
    """

    formatted_mapping = mapping_df.select(
        "id",
        F.col("normalization_struct.normalized_categories")
        .cast(T.ArrayType(T.StringType()))
        .alias("normalized_categories"),
    )

    nodes_for_join = nodes.select(
        "id", "original_id", "normalization_success", "original_categories", "all_categories"
    ).join(formatted_mapping, on="id", how="left")

    return (
        nodes_for_join.withColumn("source_role", F.lit("node"))
        .withColumn("upstream_data_source", F.lit(source))
        .select(
            "id",
            "original_id",
            "normalization_success",
            "original_categories",
            "normalized_categories",
            "all_categories",
            "source_role",
            "upstream_data_source",
        )
    )
