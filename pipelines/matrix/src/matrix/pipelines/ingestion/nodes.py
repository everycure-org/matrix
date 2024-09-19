"""Ingest nodes."""

import pyspark.sql.functions as f

from pyspark.sql import DataFrame


def ingest_robokop_nodes(raw_nodes: DataFrame) -> DataFrame:
    """Ingest Robokop nodes.

    Args:
        raw_nodes: Raw Robokop nodes.

    Returns:
        Processed Robokop nodes for int layer.
    """
    return raw_nodes.withColumn("kg_source", f.lit("robokop")).withColumn(
        "category", f.split("category", "\x1f")
    )
