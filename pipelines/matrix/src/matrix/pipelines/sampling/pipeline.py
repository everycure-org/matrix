"""Sampling pipeline."""

from kedro.pipeline import Pipeline, pipeline, node
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create a pipeline that samples datasets."""
    return pipeline(
        [
            node(
                func=nodes.sample_datasets,
                inputs={
                    "robokop_nodes": "ingestion.raw.robokop.nodes@spark",
                    "robokop_edges": "ingestion.raw.robokop.edges@spark",
                    "rtx_kg2_nodes": "ingestion.raw.rtx_kg2.nodes@spark",
                    "rtx_kg2_edges": "ingestion.raw.rtx_kg2.edges@spark",
                },
                outputs=[
                    "sampling.raw.robokop.nodes@spark",
                    "sampling.raw.robokop.edges@spark",
                    "sampling.raw.rtx_kg2.nodes@spark",
                    "sampling.raw.rtx_kg2.edges@spark",
                ],
                name="sample_datasets",
            ),
        ]
    )
