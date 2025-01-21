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
                    "integration.int.rtx_kg2.original.nodes.norm@spark",
                    "integration.int.rtx_kg2.original.edges.norm@spark",
                ],
                outputs=[
                    "integration.int.rtx_kg2.nodes.norm@spark",
                    "integration.int.rtx_kg2.edges.norm@spark",
                ],
                name="sample",
            )
        ]
    )
