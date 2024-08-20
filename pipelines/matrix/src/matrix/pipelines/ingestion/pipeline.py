"""Ingestion pipeline."""
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.prm.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.prm.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
            ),

            node(
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.prm.robokop.nodes",
                name="write_robokop_nodes",
                tags=["robokop"],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.prm.robokop.edges",
                name="write_robokop_edges",
                tags=["robokop"],
            ),

            # Run using kedro run -p ingestion -t robokop
        ]
    )
