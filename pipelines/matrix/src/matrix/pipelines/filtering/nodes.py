import logging
from typing import Any

import pyspark.sql as ps
import pyspark.sql.functions as F
from matrix_inject.inject import inject_object

from matrix.pipelines.filtering.filters import Filter

logger = logging.getLogger(__name__)


def _apply_transformations(
    df: ps.DataFrame, filters: dict[str, Filter], key_cols: list[str]
) -> tuple[ps.DataFrame, ps.DataFrame]:
    """Apply a series of filters to a DataFrame and track removed rows.

    This function applies a sequence of filters to a DataFrame while maintaining
    logging information about the number of rows removed at each step. It also
    tracks which rows were removed by comparing the original and final DataFrames
    using the specified key columns.

    Args:
        df: Input DataFrame to filter
        filters: Dictionary of filter instances to apply, where the key is the filter name
        key_cols: List of columns to use for identifying removed rows

    Returns:
        Tuple of (filtered DataFrame, DataFrame containing removed rows)
    """
    logger.info(f"Filtering dataframe with {len(filters)} filters")
    if logger.isEnabledFor(logging.INFO):
        last_count = df.count()
        logger.info(f"Number of rows before filtering: {last_count}")

    original_df = df
    for name, filter_instance in filters.items():
        logger.info(f"Applying filter: {name}")
        df_new = filter_instance.apply(df)

        if logger.isEnabledFor(logging.INFO):
            # Spark optimization with memory constraints:
            # If you really want to log after every transformation,
            # make sure to cache the new frame before the action.
            # Also, unpersist any previous cached dataframes, so we keep the memory consumption lower.
            new_count = df_new.cache().count()
            df.unpersist()
            logger.info(
                f"Number of rows after transformation '{name}': {new_count}, removed {last_count - new_count} rows"
            )
            last_count = new_count
        df = df_new

    # Identify removed rows based on key columns
    removed_df = original_df.select(*key_cols).subtract(df.select(*key_cols))
    removed_full = original_df.join(removed_df, on=key_cols, how="inner")

    return df, removed_full


@inject_object()
def prefilter_unified_kg_nodes(
    nodes: ps.DataFrame,
    transformations: dict[str, Any],
) -> tuple[ps.DataFrame, ps.DataFrame]:
    """Filter nodes using the configured filters.

    Args:
        nodes: DataFrame containing node data
        transformations: Dictionary containing filter configurations

    Returns:
        Tuple of (filtered nodes DataFrame, removed nodes DataFrame)
    """
    logger.info(f"Applying {len(transformations)} node filters")
    return _apply_transformations(nodes, transformations, key_cols=["id"])


@inject_object()
def filter_unified_kg_edges(
    nodes: ps.DataFrame,
    edges: ps.DataFrame,
    transformations: dict[str, Any],
) -> tuple[ps.DataFrame, ps.DataFrame]:
    """Function to filter the knowledge graph edges.

    We first apply a series for filter transformations, and then deduplicate the edges based on the nodes that we dropped.
    No edge can exist without its nodes.

    Args:
        nodes: DataFrame containing node data
        edges: DataFrame containing edge data
        transformations: Dictionary containing filter configurations

    Returns:
        Tuple of (filtered edges DataFrame, removed edges DataFrame)
    """
    # filter down edges to only include those that are present in the filtered nodes
    if logger.isEnabledFor(logging.INFO):
        edges_count = edges.count()
        logger.info(f"Number of edges before filtering: {edges_count}")
    original_edges = edges
    edges = (
        edges.alias("edges")
        .join(nodes.alias("subject"), on=F.col("edges.subject") == F.col("subject.id"), how="inner")
        .join(nodes.alias("object"), on=F.col("edges.object") == F.col("object.id"), how="inner")
        .select(
            "edges.*",
            F.col("subject.category").alias("subject_category"),
            F.col("object.category").alias("object_category"),
        )
    )

    logger.info(f"Applying {len(transformations)} edge filters")
    return _apply_transformations(edges, transformations, key_cols=["subject", "predicate", "object"])


def filter_nodes_without_edges(
    nodes: ps.DataFrame,
    edges: ps.DataFrame,
) -> tuple[ps.DataFrame, ps.DataFrame]:
    """Function to filter nodes without edges.

    Args:
        nodes: nodes df
        edges: edge df
    Returns:
        Final dataframe of nodes with edges
    """
    # Construct list of edges
    if logger.isEnabledFor(logging.INFO):
        logger.info("Nodes before filtering: %s", nodes.count())

    edge_nodes = (
        edges.withColumn("id", F.col("subject"))
        .unionByName(edges.withColumn("id", F.col("object")))
        .select("id")
        .distinct()
    )

    nodes_with_edges = nodes.alias("nodes").join(edge_nodes, on="id").select("nodes.*")

    removed_node_ids = nodes.select("id").subtract(nodes_with_edges.select("id"))
    removed_nodes = nodes.join(removed_node_ids, on="id", how="inner")

    if logger.isEnabledFor(logging.INFO):
        logger.info("Nodes after filtering: %s", nodes.cache().count())
    return nodes_with_edges, removed_nodes
