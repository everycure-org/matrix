"""Integration pipeline."""
from datetime import datetime

from kedro.pipeline import Pipeline, node, pipeline
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def to_neo4j(raw_nodes: DataFrame) -> DataFrame:
    """Function to create int embeddings."""
    return raw_nodes


def apply_date_filter(drugs: DataFrame, cutoff_date: datetime) -> None:
    """Function to create int embeddings."""
    return drugs.filter(F.col("date_discovered") <= cutoff_date)


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=to_neo4j,
                inputs=["integration.raw.rtx_kg2.nodes"],
                outputs="integration.prm.drugs",
                name="create_neo4j_drug_nodes",
            ),
            node(
                func=apply_date_filter,
                inputs=["integration.prm.drugs", "params:integration.cutoff_date"],
                outputs="integration.prm.filtered_drugs",
                name="filter_neo4j_drug_nodes",
            ),
        ]
    )
