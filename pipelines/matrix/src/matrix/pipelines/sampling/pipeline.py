import logging

from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    pipelines = [
        pipeline(
            [
                ArgoNode(
                    func=nodes.sample_nodes,
                    inputs=[
                        "params:sampling.configuration.sampler",
                        "integration.prm.original_filtered_nodes",
                        "integration.prm.original_filtered_edges",
                        "modelling.raw.ground_truth.original_positives@spark",
                        "modelling.raw.ground_truth.original_negatives@spark",
                        "embeddings.feat.original_nodes",
                    ],
                    outputs=[
                        "integration.prm.filtered_nodes",
                        "modelling.raw.ground_truth.positives",
                        "modelling.raw.ground_truth.negatives",
                        "embeddings.feat.nodes",
                    ],
                    name="sample",
                )
            ]
        )
    ]

    return sum(pipelines)
