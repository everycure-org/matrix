import pandas as pd
import pyspark.sql.functions as F

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """Create KG sample pipeline."""
    return pipeline(
        [
            # Takes a sample from RTX
            node(
                func=nodes.get_random_selection_from_rtx,
                inputs={
                    "nodes" : "ingestion.pre_sample.rtx_kg2.nodes@spark",
                    "edges" : "ingestion.pre_sample.rtx_kg2.edges@spark",
                },
                outputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@spark",
                    "edges": "ingestion.raw.rtx_kg2.edges@spark",
                },
                name="prefilter_rtx_kg2_nodes",
            ),
        ]
    )


