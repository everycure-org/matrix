"""Ingestion pipeline."""
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs=["integration.raw.rtx_kg2.nodes@spark"],
                outputs="integration.prm.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2", "first_copy"],
            ),
            node(
                func=lambda x: x,
                inputs=["integration.raw.rtx_kg2.edges@spark"],
                outputs="integration.prm.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2", "first_copy"],
            ),
        ]
    )
