import pandas as pd


from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """Create KG sample pipeline."""
    return pipeline(
        [
            node(
                func=nodes.get_random_selection_of_edges,
                inputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@spark",
                    "edges": "ingestion.raw.rtx_kg2.edges@spark",
                },
                outputs={
                    "nodes": "ingestion.int.sample.rtx_kg2.nodes",
                    "edges": "ingestion.int.sample.rtx_kg2.edges",
                },
                name="sampled_kg2_datasets",
            ),
        ]
    )


