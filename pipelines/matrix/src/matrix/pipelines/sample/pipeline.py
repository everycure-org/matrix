import logging

import pyspark.sql as ps
from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    pipelines = [
        pipeline(
            [
                ArgoNode(
                    func=nodes.sample_kg,
                    inputs={
                        "nodes": "integration.prm.original_filtered_nodes",
                        "edges": "integration.prm.original_filtered_edges",
                    },
                    outputs=[
                        "integration.prm.sampled_nodes",
                        "integration.prm.sampled_edges",
                    ],
                    name="sample_kg",
                )
            ]
        )
    ]

    return sum(pipelines)
