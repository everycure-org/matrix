"""Nodes for the ingration pipeline."""

from functools import partial, reduce

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key

from matrix.schemas.knowledge_graph import KGEdgeSchema, KGNodeSchema


def unify_edges(*edges) -> DataFrame:
    """Function to unify edges datasets."""
    # Union edges
    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), edges)

    # Deduplicate
    return KGEdgeSchema.group_edges_by_id(union)


def unify_nodes(*nodes) -> DataFrame:
    """Function to unify nodes datasets."""
    # Union nodes

    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), nodes)

    return KGNodeSchema.group_nodes_by_id(union)


@has_schema(
    schema={
        "label": "string",
        "source_id": "string",
        "target_id": "string",
        "property_keys": "array<string>",
        "property_values": "array<numeric>",
    },
    allow_subset=True,
)
@primary_key(primary_key=["source_id", "target_id", "label"])
def create_treats(nodes: DataFrame, df: DataFrame):
    """Function to construct treats relatonship.

    NOTE: This function requires the nodes dataset, as the nodes should be
    written _prior_ to the relationships.

    Args:
        nodes: nodes dataset
        df: Ground truth dataset
    """
    return (
        df.withColumn(
            "label", F.when(F.col("y") == 1, "TREATS").otherwise("NOT_TREATS")
        )
        .withColumn(
            "properties",
            F.create_map(
                F.lit("treats"),
                F.col("y"),
            ),
        )
        .withColumn("source_id", F.col("source"))
        .withColumn("target_id", F.col("target"))
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )
