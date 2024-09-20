"""Integration pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes
from .standardize import (
    transform_robo_nodes,
    transform_robo_edges,
    transform_rtxkg2_nodes,
    transform_rtxkg2_edges,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=transform_robo_nodes,
                inputs="ingestion.int.robokop.nodes",
                outputs="integration.prm.robokop.nodes",
                name="transform_robokop_nodes",
                tags=["standardize"],
            ),
            node(
                func=transform_robo_edges,
                inputs="ingestion.int.robokop.edges",
                outputs="integration.prm.robokop.edges",
                name="transform_robokop_edges",
                tags=["standardize"],
            ),
            node(
                func=transform_rtxkg2_nodes,
                inputs="ingestion.int.rtx_kg2.nodes",
                outputs="integration.prm.rtx.nodes",
                name="transform_rtx_nodes",
                tags=["standardize"],
            ),
            node(
                func=transform_rtxkg2_edges,
                inputs="ingestion.int.rtx_kg2.edges",
                outputs="integration.prm.rtx.edges",
                name="transform_rtx_edges",
                tags=["standardize"],
            ),
            node(
                func=nodes.unify_nodes,
                inputs=[
                    "ingestion.int.rtx_kg2.nodes",
                    "ingestion.int.ec_medical_team.nodes",
                ],
                outputs="integration.prm.unified_nodes",
                name="create_prm_unified_nodes",
            ),
            # union edges
            node(
                func=nodes.unify_edges,
                inputs=[
                    "ingestion.int.rtx_kg2.edges",
                    "ingestion.int.ec_medical_team.edges",
                ],
                outputs="integration.prm.unified_edges",
                name="create_prm_unified_edges",
            ),
        ]
    )
