"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Write kg2 Neo4J
            # Not needed until later
            # node(
            #     func=nodes.create_edges,
            #     inputs=[
            #         "integration.model_input.nodes",
            #         "ingestion.prm.rtx_kg2.edges",
            #         "params:integration.graphsage_excl_preds",
            #     ],
            #     outputs="integration.model_input.edges",
            #     name="create_neo4j_edges",
            #     tags=["rtx_kg2", "neo4j"],
            # ),
            # Construct ground_truth
            # FUTURE: Move to ground truth pipeline
            node(
                func=nodes.create_int_pairs,
                inputs=[
                    "integration.raw.ground_truth.positives",
                    "integration.raw.ground_truth.negatives",
                ],
                outputs="integration.int.known_pairs@pandas",
                name="create_int_known_pairs",
                tags=["rtx_kg2", "neo4j", "first_copy"],
            ),
            node(
                func=nodes.create_treats,
                inputs=[
                    "integration.model_input.nodes",
                    "integration.int.known_pairs@spark",
                ],
                outputs="integration.model_input.ground_truth",
                name="create_neo4j_known_pairs",
                tags=["rtx_kg2", "neo4j"],
            ),
        ]
    )
