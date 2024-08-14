"""Integration pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # union edges
            node(
                func=nodes.unify_edges,
                inputs=["ingestion.prm.rtx_kg2.edges", "preprocessing.prm.edges@spark"],
                outputs="integration.prm.edges",
                name="create_prm_edges",
                tags=["rtx_kg2"],
            ),
            # Write kg2 Neo4J
            node(
                func=nodes.create_nodes,
                inputs=["ingestion.prm.rtx_kg2.nodes"],
                outputs="integration.model_input.nodes",
                name="create_neo4j_nodes",
                tags=["rtx_kg2", "neo4j"],
            ),
            node(
                func=nodes.create_edges,
                inputs=[
                    "integration.model_input.nodes",
                    "integration.prm.edges",
                    "params:integration.graphsage_excl_preds",
                ],
                outputs="integration.model_input.edges",
                name="create_neo4j_edges",
                tags=["rtx_kg2", "neo4j"],
            ),
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
