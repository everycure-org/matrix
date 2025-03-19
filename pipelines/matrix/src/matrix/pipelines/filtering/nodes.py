import logging
from typing import Any, Callable, Dict, List, Tuple

import pyspark.sql as ps
import pyspark.sql.functions as F

from matrix.inject import inject_object

logger = logging.getLogger(__name__)


def _apply_transformations(
    df: ps.DataFrame, transformations: List[Tuple[Callable, Dict[str, Any]]], **kwargs
) -> ps.DataFrame:
    logger.info(f"Filtering dataframe with {len(transformations)} transformations")
    if logger.isEnabledFor(logging.INFO):
        last_count = df.count()
        logger.info(f"Number of rows before filtering: {last_count}")
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

    return df


@inject_object()
def prefilter_unified_kg_nodes(
    nodes: ps.DataFrame,
    transformations: List[Tuple[Callable, Dict[str, Any]]],
) -> ps.DataFrame:
    return _apply_transformations(nodes, transformations)


@inject_object()
def filter_unified_kg_edges(
    nodes: ps.DataFrame,
    edges: ps.DataFrame,
    transformations: List[Tuple[Callable, Dict[str, Any]]],
) -> ps.DataFrame:
    """Function to filter the knowledge graph edges.

    We first apply a series for filter transformations, and then deduplicate the edges based on the nodes that we dropped.
    No edge can exist without its nodes.
    """

    # filter down edges to only include those that are present in the filtered nodes
    if logger.isEnabledFor(logging.INFO):
        edges_count = edges.count()
        logger.info(f"Number of edges before filtering: {edges_count}")
    edges = (
        edges.alias("edges")
        .join(nodes.alias("subject"), on=F.col("edges.subject") == F.col("subject.id"), how="inner")
        .join(nodes.alias("object"), on=F.col("edges.object") == F.col("object.id"), how="inner")
        .select("edges.*")
    )
    if logger.isEnabledFor(logging.INFO):
        new_edges_count = edges.cache().count()
        logger.info(
            f"Number of edges after filtering: {new_edges_count}, cut out {edges_count - new_edges_count} edges"
        )

    return _apply_transformations(edges, transformations)


def filter_nodes_without_edges(
    nodes: ps.DataFrame,
    edges: ps.DataFrame,
) -> ps.DataFrame:
    """Function to filter nodes without edges.

    Args:
        nodes: nodes df
        edges: edge df
    Returns"
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

    nodes = nodes.alias("nodes").join(edge_nodes, on="id").select("nodes.*")
    if logger.isEnabledFor(logging.INFO):
        logger.info("Nodes after filtering: %s", nodes.cache().count())
    return nodes
