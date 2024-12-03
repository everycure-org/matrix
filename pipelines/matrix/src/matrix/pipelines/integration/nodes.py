import logging
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import pandera.pyspark as pa
import pyspark as ps
import pyspark.sql.functions as F
from joblib import Memory
from pyspark.sql import DataFrame
from refit.v1.core.inject import inject_object

from matrix.pipelines.integration.filters import determine_most_specific_category
from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema, cols_for_schema

# TODO move these into config
memory = Memory(location=".cache/nodenorm", verbose=0)
logger = logging.getLogger(__name__)


@pa.check_output(KGEdgeSchema)
def union_and_deduplicate_edges(*edges) -> DataFrame:
    """Function to unify edges datasets."""
    # fmt: off
    return (
        _union_datasets(*edges)
        .transform(KGEdgeSchema.group_edges_by_id)
    )
    # fmt: on


@pa.check_output(KGNodeSchema)
def union_and_deduplicate_nodes(biolink_categories_df: pd.DataFrame, *nodes) -> DataFrame:
    """Function to unify nodes datasets."""

    # fmt: off
    return (
        _union_datasets(*nodes)

        # first we group the dataset by id to deduplicate
        .transform(KGNodeSchema.group_nodes_by_id)

        # next we need to apply a number of transformations to the nodes to ensure grouping by id did not select wrong information
        .transform(determine_most_specific_category, biolink_categories_df)

        # finally we select the columns that we want to keep
        .select(*cols_for_schema(KGNodeSchema))
    )
    # fmt: on


def _union_datasets(
    *datasets: DataFrame,
) -> DataFrame:
    """
    Helper function to unify datasets and deduplicate them.

    Args:
        datasets_to_union: List of dataset names to unify.
        **datasets: Arbitrary number of DataFrame keyword arguments.
        schema_group_by_id: Function to deduplicate the unified DataFrame.

    Returns:
        A unified and deduplicated DataFrame.
    """
    return reduce(partial(DataFrame.unionByName, allowMissingColumns=True), datasets)


def _apply_transformations(
    df: DataFrame, transformations: List[Tuple[Callable, Dict[str, Any]]], **kwargs
) -> DataFrame:
    logger.info(f"Filtering dataframe with {len(transformations)} transformations")
    last_count = df.count()
    logger.info(f"Number of rows before filtering: {last_count}")
    for name, transformation in transformations.items():
        logger.info(f"Applying transformation: {name}")
        df = df.transform(transformation, **kwargs)
        new_count = df.count()
        logger.info(f"Number of rows after transformation: {new_count}, cut out {last_count - new_count} rows")
        last_count = new_count

    return df


@inject_object()
def prefilter_unified_kg_nodes(
    nodes: DataFrame,
    transformations: List[Tuple[Callable, Dict[str, Any]]],
) -> DataFrame:
    return _apply_transformations(nodes, transformations)


@inject_object()
def filter_unified_kg_edges(
    nodes: DataFrame,
    edges: DataFrame,
    biolink_predicates: Dict[str, Any],
    transformations: List[Tuple[Callable, Dict[str, Any]]],
) -> DataFrame:
    """Function to filter the knowledge graph edges.

    We first apply a series for filter transformations, and then deduplicate the edges based on the nodes that we dropped.
    No edge can exist without its nodes.
    """

    # filter down edges to only include those that are present in the filtered nodes
    edges_count = edges.count()
    logger.info(f"Number of edges before filtering: {edges_count}")
    edges = (
        edges.alias("edges")
        .join(nodes.alias("subject"), on=F.col("edges.subject") == F.col("subject.id"), how="inner")
        .join(nodes.alias("object"), on=F.col("edges.object") == F.col("object.id"), how="inner")
        .select("edges.*")
    )
    new_edges_count = edges.count()
    logger.info(f"Number of edges after filtering: {new_edges_count}, cut out {edges_count - new_edges_count} edges")

    return _apply_transformations(edges, transformations, biolink_predicates=biolink_predicates)


def filter_nodes_without_edges(
    nodes: DataFrame,
    edges: DataFrame,
) -> DataFrame:
    """Function to filter nodes without edges.

    Args:
        nodes: nodes df
        edges: edge df
    Returns"
        Final dataframe of nodes with edges
    """

    # Construct list of edges
    logger.info("Nodes before filtering: %s", nodes.count())
    edge_nodes = (
        edges.withColumn("id", F.col("subject"))
        .unionByName(edges.withColumn("id", F.col("object")))
        .select("id")
        .distinct()
    )

    nodes = nodes.alias("nodes").join(edge_nodes, on="id").select("nodes.*").persist()
    logger.info("Nodes after filtering: %s", nodes.count())
    return nodes


def normalize_kg(
    mapping_df: ps.sql.DataFrame,
    nodes: ps.sql.DataFrame,
    edges: ps.sql.DataFrame,
) -> ps.sql.DataFrame:
    """Function normalizes a KG using external API endpoint.

    This function takes the nodes and edges frames for a KG and leverages
    an external API to map the nodes to their normalized IDs.
    It returns the datasets with normalized IDs.

    """
    mapping_df = (
        mapping_df.drop("bucket")
        .withColumn("normalization_success", F.col("normalized_id").isNotNull())
        # avoids nulls in id column, if we couldn't resolve IDs, we keep original
        .withColumn("normalized_id", F.coalesce(F.col("normalized_id"), F.col("id")))
    )

    # add normalized_id to nodes
    nodes = (
        nodes.join(mapping_df, on="id", how="left")
        .withColumnsRenamed({"id": "original_id"})
        .withColumnsRenamed({"normalized_id": "id"})
    )

    # edges are bit more complex, we need to map both the subject and object
    edges = edges.join(
        mapping_df.withColumnsRenamed(
            {
                "id": "subject",
                "normalized_id": "subject_normalized",
                "normalization_success": "subject_normalization_success",
            }
        ),
        on="subject",
        how="left",
    )
    edges = edges.join(
        mapping_df.withColumnsRenamed(
            {
                "id": "object",
                "normalized_id": "object_normalized",
                "normalization_success": "object_normalization_success",
            }
        ),
        on="object",
        how="left",
    )
    edges = edges.withColumnsRenamed({"subject": "original_subject", "object": "original_object"}).withColumnsRenamed(
        {"subject_normalized": "subject", "object_normalized": "object"}
    )

    return nodes, edges  # mapping_df
