"""Embeddings pipeline."""
from typing import List

from kedro.pipeline import Pipeline, node, pipeline

import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.feature import Word2Vec

def print_(df: DataFrame):
    df.show()
    return df

def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # NOTE: This enriches the current graph with a nummeric property
            node(
                func=print_,
                inputs=["integration.model_input.nodes"],
                outputs="embeddings.prm.graph",
                name="enrich_graph",
            ),
            # node(
            #     func=print_,
            #     inputs=["embeddings.prm.graph", "embeddings.model_input.gds.graph"],
            #     outputs="_graph",
            #     name="create_gds_graph",
            # ),
            # node(
            #     func=print_,
            #     inputs=["_graph", "embeddings.models.gds.graphsage"],
            #     outputs="_model",
            #     name="train_graphsage_embeddings",
            # ),
            # node(
            #     func=print_,
            #     inputs=["_model", "embeddings.model_output.gds.graphsage"],
            #     outputs="embeddings.model_output.graphsage",
            #     name="create_graphsage_embeddings",
            # ),
        ]
    )
