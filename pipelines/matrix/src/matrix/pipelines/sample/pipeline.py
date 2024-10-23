import pandas as pd
import pyspark.sql.functions as F

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """Create KG sample pipeline."""
    return pipeline(
        [
            # Copy datasets locally
            node(
                func=lambda x: x,
                inputs="ingestion.pre_sample.rtx_kg2.nodes@spark",
                outputs="sample.int.rtx-nodes",
                name="copy-rtx-nodes",
                tags=["copy"]
            ),
            node(
                func=lambda x: x,
                inputs="ingestion.pre_sample.rtx_kg2.edges@spark",
                outputs="sample.int.rtx-edges",
                name="copy-rtx-edges",
                tags=["copy"]
            ),
            node(
                func=nodes.get_random_selection_from_rtx,
                inputs={
                    "nodes" : "sample.int.rtx-nodes",
                    "edges" : "sample.int.rtx-edges",
                },
                outputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@spark",
                    "edges": "ingestion.raw.rtx_kg2.edges@spark",
                },
                name="prefilter_rtx_kg2",
                tags=["sample"]
            ),
        ]
    )


