"""Embeddings pipeline."""
import os
from typing import List, Any, Dict

from kedro.pipeline import Pipeline, node, pipeline

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.functions import array_to_vector, vector_to_array
from graphdatascience import GraphDataScience

from refit.v1.core.inject import inject_object
from refit.v1.core.unpack import unpack_params

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # NOTE: This enriches the current graph with a nummeric property
            node(
                func=nodes.concat_features,
                inputs=[
                    "integration.model_input.nodes",
                    "params:embeddings.node.features",
                    "params:embeddings.ai_config",
                ],
                outputs="embeddings.prm.graph.embeddings",
                name="add_node_embeddings",
            ),
            node(
                func=nodes.reduce_dimension,
                inputs=[
                    "embeddings.prm.graph.embeddings",
                    "params:embeddings.dimensionality_reduction",
                ],
                outputs="embeddings.prm.graph.pca_embeddings",
                name="apply_pca",
            ),
            node(
                func=nodes.add_topological_embeddings,
                inputs={
                    "df": "embeddings.prm.graph.pca_embeddings",
                    "edges": "integration.model_input.edges",
                    "unpack": "params:embeddings.topological",
                },
                outputs="embeddings.model_output.graphsage",  # TODO: Add dummy dataset
                name="add_topological_embeddings",
            ),
        ]
    )
