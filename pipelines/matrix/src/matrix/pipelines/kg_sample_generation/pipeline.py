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
                    "nodes": "ingestion.int.rtx_kg2.nodes",
                    "edges": "ingestion.int.rtx_kg2.edges",
                },
                outputs={
                    "nodes": "ingestion.sample.rtx_kg2.nodes",
                    "edges": "ingestion.sample.rtx_kg2.edges",
                },
                name="sampled_kg2_datasets",
            ),
            # ec-medical-team
            node(
                func=lambda x: x,
                inputs=["ingestion.int.ec_medical_team.nodes"],
                outputs="ingestion.sample.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=["ec_medical_team"],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.int.ec_medical_team.edges"],
                outputs="ingestion.sample.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=["ec_medical_team"],
            ),
        ]
    )


