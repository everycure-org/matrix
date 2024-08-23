"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=nodes.unify_nodes,
                inputs=[
                    "ingestion.prm.rtx_kg2.nodes",
                    "ingestion.prm.ec_medical_team.nodes",
                ],
                outputs="integration.prm.unified_nodes",
                name="create_prm_unified_nodes",
            ),
            # union edges
            node(
                func=nodes.unify_edges,
                inputs=[
                    "ingestion.prm.rtx_kg2.edges",
                    "ingestion.prm.ec_medical_team.edges",
                ],
                outputs="integration.prm.unified_edges",
                name="create_prm_unified_edges",
            ),
            # TODO: Needs a fix
            # # Construct ground_truth
            # node(
            #     func=nodes.create_int_pairs,
            #     inputs=[
            #         "integration.raw.ground_truth.positives",
            #         "integration.raw.ground_truth.negatives",
            #     ],
            #     outputs="integration.int.known_pairs@pandas",
            #     name="create_int_known_pairs",
            #     tags=["rtx_kg2", "neo4j", "first_copy"],
            # ),
            # node(
            #     func=nodes.create_treats,
            #     inputs=[
            #         "integration.model_input.nodes",
            #         "integration.int.known_pairs@spark",
            #     ],
            #     outputs="integration.model_input.ground_truth",
            #     name="create_neo4j_known_pairs",
            #     tags=["rtx_kg2", "neo4j"],
            # ),
        ]
    )
