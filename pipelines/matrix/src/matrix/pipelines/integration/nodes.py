"""Nodes for the ingration pipeline."""
import pandas as pd
from functools import reduce, partial

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key


def unify_edges(*edges) -> DataFrame:
    """Function to unify edges datasets."""
    # Union edges
    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), edges)

    # Deduplicate
    return union.groupBy(["subject", "predicate", "object"]).agg(
        F.collect_list("knowledge_source").alias("knowledge_sources"),
        F.collect_list("kg_source").alias("kg_sources"),
    )


def unify_nodes(*nodes) -> DataFrame:
    """Function to unify nodes datasets."""
    # Union nodes

    union = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), nodes)

    # Deduplicate
    # FUTURE: We should improve selection of name and description currently
    # selecting the first non-null, which might not be as desired.
    return union.groupBy(["id"]).agg(
        F.collect_list("kg_source").alias("kg_sources"),
        F.first("name").alias("name"),
        F.first("description").alias("description"),
        F.first("category").alias("category"),
    )


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
