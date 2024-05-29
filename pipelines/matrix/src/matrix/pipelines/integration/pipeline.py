"""Embeddings pipeline."""
import pandas as pd

from typing import List

from refit.v1.core.inline_has_schema import has_schema

from kedro.pipeline import Pipeline, node, pipeline
from pyspark.sql import DataFrame


def to_neo4j(raw_nodes: DataFrame) -> DataFrame:
    """Function to create int embeddings."""
    return raw_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=to_neo4j,
                inputs=["integration.raw.rtx_kg2.nodes"],
                outputs="integration.prm.neo4j_nodes",
                name="create_neo4j_nodes",
            ),
        ]
    )
