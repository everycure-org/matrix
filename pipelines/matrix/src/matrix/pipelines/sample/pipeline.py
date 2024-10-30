import pandas as pd
import pyspark.sql.functions as F

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """Create KG sample pipeline."""
    return pipeline(
        [
            # Make sample and save files in sample directory
            node(
                func=nodes.get_random_sample_from_unified_files,
                inputs={
                    "nodes" : "integration.prm_ingest.unified_nodes",
                    "edges" : "integration.prm_ingest.unified_edges",
                },
                outputs={
                    "nodes" : "integration.prm.unified_nodes",
                    "edges" : "integration.prm.unified_edges",
                },
                name="sample_unified_files",
                tags=["sample"]
            ),
        ]
    )





