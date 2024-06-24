"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Write kg2
            node(
                func=lambda x: x,
                inputs=["integration.raw.rtx_kg2.nodes@spark"],
                outputs="integration.prm.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            # Write Neo4J nodes
            node(
                func=nodes.create_nodes,
                inputs=["integration.prm.rtx_kg2.nodes"],
                outputs="integration.model_input.nodes",
                name="create_neo4j_nodes",
                tags=["rtx_kg2"],
            ),
            # Construct ground_truth
            node(
                func=nodes.create_int_pairs,
                inputs=[
                    "integration.raw.ground_truth.tp",
                    "integration.raw.ground_truth.tn",
                ],
                outputs="integration.int.known_pairs@pandas",
                name="create_int_known_pairs",
            ),
            node(
                func=nodes.create_treats,
                inputs=[
                    "integration.model_input.nodes",
                    "integration.int.known_pairs@spark",
                ],
                outputs="integration.model_input.treats",
                name="create_neo4j_known_pairs",
            ),
        ]
    )
