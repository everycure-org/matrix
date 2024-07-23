"""Nodes for the ingration pipeline."""
import pandas as pd

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


@has_schema(
    schema={
        "subject": "string",
        "predicate": "string",
        "object": "string",
    },
    allow_subset=True,
)
def write_edges(edges: DataFrame):
    """Function to filter out treat and not treat edges and write.

    Args:
        edges: edges dataframe
    """
    return edges.filter(~edges["predicate"].rlike("(?i)treats"))


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
        df.select("id", "category", "name", "description")
        .withColumn("label", F.split(F.col("category"), ":", limit=2).getItem(1))
        .withColumn(
            "properties",
            F.create_map(
                F.lit("name"),
                F.col("name"),
                F.lit("description"),
                F.col("description"),
                F.lit("category"),
                F.col("category"),
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
    },
    allow_subset=True,
)
def create_edges(nodes: DataFrame, edges: DataFrame):
    """Function to create Neo4J edges.

    Args:
        nodes: nodes dataframe
        edges: edges dataframe
    """
    return edges.select(
        "subject", "predicate", "object", "knowledge_source"
    ).withColumn("label", F.split(F.col("predicate"), ":", limit=2).getItem(1))


@has_schema(
    schema={
        "label": "string",
        "source_id": "string",
        "target_id": "string",
        "property_keys": "array<string>",
        "property_values": "array<string>",
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
            F.create_map(F.lit("treats"), F.col("y"), F.lit("foo"), F.lit("bar")),
        )
        .withColumn("source_id", F.col("source"))
        .withColumn("target_id", F.col("target"))
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )
