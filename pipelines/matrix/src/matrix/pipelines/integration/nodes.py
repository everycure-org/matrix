"""Nodes for the ingration pipeline."""
import pandas as pd
from typing import List
from functools import reduce, partial

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key


def create_int_pairs(raw_tp: pd.DataFrame, raw_tn: pd.DataFrame):
    """Create intermediate pairs dataset.

    Args:
        raw_tp: Raw ground truth positive data.
        raw_tn: Raw ground truth negative data.

    Returns:
        Combined ground truth positive and negative data.
    """
    raw_tp["y"] = 1
    raw_tn["y"] = 0

    # Concat
    return pd.concat([raw_tp, raw_tn], axis="index").reset_index(drop=True)


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
        "id": "string",
        "name": "string",
        "property_keys": "array<string>",
        "property_values": "array<string>",
    },
    allow_subset=True,
)
@primary_key(primary_key=["id"])
def create_nodes(df: DataFrame) -> DataFrame:
    """Function to create Neo4J nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df.select("id", "name", "category", "description", "kg_sources")
        .withColumn("label", F.split(F.col("category"), ":", limit=2).getItem(1))
        .withColumn(
            "properties",
            F.create_map(
                F.lit("name"),
                F.col("name"),
                F.lit("category"),
                F.col("category"),
                F.lit("description"),
                F.col("description"),
            ),
        )
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )


@has_schema(
    schema={
        "subject": "string",
        "predicate": "string",
        "object": "string",
        "label": "string",
    },
    allow_subset=True,
)
def create_edges(nodes: DataFrame, edges: DataFrame, exc_preds: List[str]):
    """Function to create Neo4J edges.

    Args:
        nodes: nodes dataframe
        edges: edges dataframe
        exc_preds: list of predicates excluded downstream
    """
    return edges.select(
        "subject", "predicate", "object", "knowledge_source", "kg_sources"
    ).withColumn("label", F.split(F.col("predicate"), ":", limit=2).getItem(1))


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
