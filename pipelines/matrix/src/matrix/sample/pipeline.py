"""Preprocessing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=filter_nodes_function,
                inputs=[
                    "ingestion.pre_sample.rtx_kg2.nodes@spark",
                    # ... and other datasets here
                ],
                outputs="ingestion.raw.rtx_kg2.nodes@spark",
                name="prefilter_rtx_kg2_nodes",
                tags=["rtx-kg2"],
            ),
            node(
                func=filter_edges_function,
                inputs=[
                    "ingestion.pre_sample.rtx_kg2.edges@spark",
                    # ... and other datasets here
                ],
                outputs="ingestion.raw.rtx_kg2.nodes@spark",
                name="prefilter_rtx_kg2_nodes",
                tags=["rtx-kg2"],
            ),
        ]
    )
