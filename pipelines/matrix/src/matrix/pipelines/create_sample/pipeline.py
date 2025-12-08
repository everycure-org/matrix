import logging

from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode

from . import nodes

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            ArgoNode(
                func=nodes.sample_knowledge_graph,
                inputs={
                    "sampler": "params:create_sample.configuration.sampler",
                    "knowledge_graph_nodes": "integration.prm.original.unified_nodes",
                    "knowledge_graph_edges": "integration.prm.original.unified_edges",
                    "ground_truth_edges": "integration.int.ground_truth.edges.norm@spark",
                },
                outputs={
                    "nodes": "integration.prm.unified_nodes@spark",
                    "edges": "integration.prm.unified_edges@spark",
                },
                name="sample",
            )
        ]
    )
