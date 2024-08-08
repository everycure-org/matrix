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
            # TODO: Add nodes to move robokop nodes and endge files to `prm`
            # TODO: Set tags to `robokop`
            # Run using kedro run -p ingestion -t robokop
        ]
    )
