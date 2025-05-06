import logging
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

import pyspark.sql as ps
import pyspark.sql.functions as F
from pyspark.sql import types as T

from matrix.inject import inject_object
from matrix.pipelines.filtering.filters import Filter, RemoveRowsByColumnFilter, TriplePatternFilter

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", bound=Filter, covariant=True)


def _create_filter_from_params(filter_name: str, filter_params: dict[str, Any]) -> Filter:
    """Create a filter instance from parameters.

    Args:
        filter_name: Name of the filter to create
        filter_params: Dictionary of parameters for the filter

    Returns:
        Instantiated filter object

    Raises:
        ValueError: If filter_name is not recognized
    """
    # Make a copy of the parameters to avoid modifying the original
    params = dict(filter_params)
    filter_type = params.pop("filter")

    if filter_type == "RemoveRowsByColumnFilter":
        return RemoveRowsByColumnFilter(**params)
    elif filter_type == "TriplePatternFilter":
        return TriplePatternFilter(**params)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def _create_filters_from_params(filters_config: dict[str, dict[str, Any]]) -> dict[str, Filter]:
    """Create filter instances from configuration.

    Args:
        filters_config: Dictionary of filter configurations

    Returns:
        Dictionary of instantiated filters
    """
    return {name: _create_filter_from_params(name, params) for name, params in filters_config.items()}


def _apply_transformations(
    df: ps.DataFrame, transformations: dict[str, Callable[[ps.DataFrame], ps.DataFrame]], key_cols: list[str], **kwargs
) -> tuple[ps.DataFrame, ps.DataFrame]:
    """Apply a series of transformations to a DataFrame and track removed rows.

    Args:
        df: Input DataFrame to transform
        transformations: Dictionary of transformation functions to apply
        key_cols: List of columns to use for identifying removed rows
        **kwargs: Additional arguments to pass to transformations

    Returns:
        Tuple of (transformed DataFrame, DataFrame containing removed rows)
    """
    logger.info(f"Filtering dataframe with {len(transformations)} transformations")
    if logger.isEnabledFor(logging.INFO):
        last_count = df.count()
        logger.info(f"Number of rows before filtering: {last_count}")
    original_df = df
    for name, transformation in transformations.items():
        logger.info(f"Applying transformation: {name}")
        df_new = df.transform(transformation, **kwargs)
        if logger.isEnabledFor(logging.INFO):
            # Spark optimization with memory constraints:
            # If you really want to log after every transformation,
            # make sure to cache the new frame before the action.
            # Also, unpersist any previous cached dataframes, so we keep the memory consumption lower.
            new_count = df_new.cache().count()
            df.unpersist()
            logger.info(f"Number of rows after transformation: {new_count}, cut out {last_count - new_count} rows")
            last_count = new_count
        df = df_new

    # Identify removed rows based on key columns
    removed_df = original_df.select(*key_cols).subtract(df.select(*key_cols))
    removed_full = original_df.join(removed_df, on=key_cols, how="inner")

    return df, removed_full


def apply_filter_transformations(
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

    Example:
        ```python
        filters = {
            "remove_drugs": RemoveRowsByColumnFilter(
                column="category",
                remove_list=["biolink:Drug"]
            ),
            "remove_drug_drug_interactions": TriplePatternFilter(
                triples_to_exclude=[
                    ["biolink:Drug", "biolink:physically_interacts_with", "biolink:Drug"]
                ]
            )
        }
        filtered_df, removed_df = apply_filter_transformations(
            df=edges_df,
            filters=filters,
            key_cols=["subject", "object", "predicate"]
        )
        ```
    """
    logger.info(f"Filtering dataframe with {len(filters)} filters")
    if logger.isEnabledFor(logging.INFO):
        last_count = df.count()
        logger.info(f"Number of rows before filtering: {last_count}")

    original_df = df
    for name, filter_instance in filters.items():
        logger.info(f"Applying filter: {name}")
        df_new = filter_instance.filter(df)

        if logger.isEnabledFor(logging.INFO):
            # Spark optimization with memory constraints:
            # Cache the new frame before the action and unpersist the old one
            new_count = df_new.cache().count()
            df.unpersist()
            logger.info(f"Number of rows after filter '{name}': {new_count}, " f"removed {last_count - new_count} rows")
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
    if transformations:
        logger.info(f"Applying {len(transformations)} node filters")
        try:
            filters = _create_filters_from_params(transformations)
            filtered, removed = apply_filter_transformations(nodes, filters, key_cols=["id"])

            if logger.isEnabledFor(logging.INFO):
                logger.info("Sample of filtered nodes:")
                logger.info(filtered.select("id", "name", "category").head(10))
                logger.info("Sample of removed nodes:")
                logger.info(removed.select("id", "name", "category").head(10))

            return filtered.select("id", "name", "category"), removed.select("id", "name", "category")
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            raise
    else:
        # Fallback to old style transformations if no filters are configured
        logger.info("No node filters configured, using legacy transformation")
        return _apply_transformations(nodes, transformations, key_cols=["id"])


@inject_object()
def filter_unified_kg_edges(
    nodes: ps.DataFrame,
    edges: ps.DataFrame,
    transformations: dict[str, Callable[[ps.DataFrame], ps.DataFrame]],
) -> tuple[ps.DataFrame, ps.DataFrame]:
    """Function to filter the knowledge graph edges.

    We first apply a series for filter transformations, and then deduplicate the edges based on the nodes that we dropped.
    No edge can exist without its nodes.
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
    if logger.isEnabledFor(logging.INFO):
        new_edges_count = edges.cache().count()
        logger.info(
            f"Number of edges after filtering: {new_edges_count}, cut out {edges_count - new_edges_count} edges"
        )

    filtered, removed = _apply_transformations(edges, transformations, key_cols=["subject", "predicate", "object"])
    # Remove new fields from schema
    filtered = filtered.drop("subject_category", "object_category")
    return filtered, removed


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
