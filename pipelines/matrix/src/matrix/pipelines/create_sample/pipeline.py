import logging

from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            ArgoNode(
                func=nodes.sample_nodes,
                inputs=[
                    "params:create_sample.configuration.sampler",
                    "integration.prm.original.filtered_nodes",
                    "integration.prm.original.filtered_edges",
                ],
                outputs=[
                    "integration.prm.filtered_nodes",
                    "integration.prm.filtered_edges",
                ],
                name="sample",
            )
        ]
    )
