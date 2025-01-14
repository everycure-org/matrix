import logging

import pyspark.sql as ps
from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.node import node

logger = logging.getLogger(__name__)


def sample_nodes(nodes: ps.DataFrame, edges: ps.DataFrame, **kwargs):
    logger.info("Yay")
    logger.info(f"Parsed {nodes.count()} nodes and {edges.count()} edges")
    raise Exception("Pwal")
    return (nodes, edges)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sample_nodes,
                inputs={
                    "nodes": "integration.prm.original_filtered_nodes",
                    "edges": "integration.prm.original_filtered_edges",
                },
                outputs=[
                    "integration.prm.sampled_nodes",
                    "integration.prm.sampled_edges",
                ],
            )
        ]
    )
