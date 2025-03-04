import logging
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
import pyspark.sql.types as T
from joblib import Memory
from pyspark.sql.window import Window

from matrix.inject import inject_object
from matrix.pipelines.integration.filters import determine_most_specific_category
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

from .schema import BIOLINK_KG_EDGE_SCHEMA, BIOLINK_KG_NODE_SCHEMA

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
        # removing the uniqueness constraint as some KGs have duplicate edges. These will be deduplicated later when we do edge deduplication anyways
        # unique=["subject", "predicate", "object"],
    ),
    df_name="edges",
    raise_df_undefined=False,
)
def transform(transformer, **kwargs) -> Dict[str, ps.DataFrame]:
    return transformer.transform(**kwargs)


@check_output(
    schema=BIOLINK_KG_EDGE_SCHEMA,
    pass_columns=True,
)
def union_edges(*edges, cols: List[str]) -> ps.DataFrame:
    """Function to unify edges datasets."""
    # fmt: off
    return (
        _union_datasets(*edges)
        .groupBy(["subject", "predicate", "object"])
        .agg(
            F.flatten(F.collect_set("upstream_data_source")).alias("upstream_data_source"),
            # TODO: we shouldn't just take the first one but collect these values from multiple upstream sources
            F.first("knowledge_level", ignorenulls=True).alias("knowledge_level"),
            F.first("subject_aspect_qualifier", ignorenulls=True).alias("subject_aspect_qualifier"),
            F.first("subject_direction_qualifier", ignorenulls=True).alias("subject_direction_qualifier"),
            F.first("object_direction_qualifier", ignorenulls=True).alias("object_direction_qualifier"),
            F.first("object_aspect_qualifier", ignorenulls=True).alias("object_aspect_qualifier"),
            F.first("primary_knowledge_source", ignorenulls=True).alias("primary_knowledge_source"),
            F.flatten(F.collect_set("aggregator_knowledge_source")).alias("aggregator_knowledge_source"),
            F.flatten(F.collect_set("publications")).alias("publications"),
        )
        .select(*cols)
    )
    # fmt: on


@check_output(
    schema=BIOLINK_KG_NODE_SCHEMA,
    pass_columns=True,
)
def union_and_deduplicate_nodes(retrieve_most_specific_category: bool, *nodes, cols: List[str]) -> ps.DataFrame:
    """Function to unify nodes datasets."""
    # fmt: off
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

    # Is this integration or filtering?
    if retrieve_most_specific_category:
        unioned_datasets = unioned_datasets.transform(determine_most_specific_category)

    return unioned_datasets.select(*cols)
    # fmt: on


def _union_datasets(
    *datasets: ps.DataFrame,
) -> ps.DataFrame:
    """
    Helper function to unify datasets and deduplicate them.
    Args:
        datasets_to_union: List of dataset names to unify.
        **datasets: Arbitrary number of DataFrame keyword arguments.
        schema_group_by_id: Function to deduplicate the unified DataFrame.

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
        mapping_df.drop("bucket")
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
    return (
        edges.withColumn(
            "_rn",
            F.row_number().over(
                Window.partitionBy(["subject", "object", "predicate"]).orderBy(F.col("original_subject"))
            ),
        )
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


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
