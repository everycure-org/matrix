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
                    func=nodes.sample,
                    inputs=[
                        "integration.prm.original_filtered_nodes",
                        "integration.prm.original_filtered_edges",
                        "modelling.raw.ground_truth.original_positives@spark",
                        "modelling.raw.ground_truth.original_negatives@spark",
                        "embeddings.feat.original_nodes",
                        "params:sampling.configuration.ground_truth_positive_sample_ratio",
                        "params:sampling.configuration.ground_truth_negative_sample_ratio",
                        "params:sampling.configuration.kg_nodes_sample_fraction",
                        "params:sampling.configuration.seed",
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
