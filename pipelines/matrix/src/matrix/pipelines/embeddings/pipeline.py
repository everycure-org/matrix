"""Embeddings pipeline."""
from typing import List

from kedro.pipeline import Pipeline, node, pipeline

import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_value(graph: DataFrame):
    # TODO: This should all features to node for embedding computation
    return graph.withColumn("feature", F.rand())


def create_graphsage_embeddings(graph: DataFrame, embeddings: DataFrame):
    embeddings.show(500)


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # NOTE: This enriches the current graph with a nummeric property
            node(
                func=add_value,
                inputs=["integration.model_input.nodes"],
                outputs="embeddings.prm.graph",
                name="enrich_graph",
            ),
            node(
                func=lambda _, y: y,
                inputs=["embeddings.prm.graph", "embeddings.model_input.gds.graph"],
                outputs="_gds_graph",
                name="create_gds_graph",
            ),
            node(
                func=lambda _, y: y,
                inputs=["_gds_graph", "embeddings.models.gds.graphsage"],
                outputs="_gds_model",
                name="train_graphsage_embeddings",
            ),
            # node(
            #     func=create_graphsage_embeddings,
            #     inputs=["_gds_model", "embeddings.model_output.gds.graphsage"],
            #     outputs=None,
            #     name="create_graphsage_embeddings"
            # )
        ]
    )
