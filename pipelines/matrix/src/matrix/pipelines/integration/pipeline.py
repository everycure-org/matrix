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
        ]
    )
