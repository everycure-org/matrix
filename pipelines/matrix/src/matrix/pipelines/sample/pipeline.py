import pandas as pd
import pyspark.sql.functions as F

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """Create KG sample pipeline."""
    return pipeline(
        [
            # Copy datasets locally (to avoid re-copying from bucket every time)
            #  1. RTX nodes and edges
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
            #  2. Ground Truths - we do the sample straight-away
            node(
                func=nodes.get_random_selection_of_ground_truths,
                inputs="modelling.pre_sample.ground_truth.positives",
                outputs="modelling.raw.ground_truth.positives",
                name="copy-ground_truths-pos",
                tags=["copy"]
            ),
            node(
                func=nodes.get_random_selection_of_ground_truths,
                inputs="modelling.pre_sample.ground_truth.negatives",
                outputs="modelling.raw.ground_truth.negatives",
                name="copy-ground_truths-neg",
                tags=["copy"]
            ),
            
            # Make sample and save files in sample directory
            node(
                func=nodes.get_random_selection_from_rtx,
                inputs={
                    "nodes" : "sample.int.rtx-nodes",
                    "edges" : "sample.int.rtx-edges",
                },
                outputs={
                    "nodes": "ingestion.int.rtx_kg2.nodes",
                    "edges": "ingestion.int.rtx_kg2.edges",
                },
                name="prefilter_rtx_kg2",
                tags=["sample"]
            ),
            # ec-medical-team
            node(
                func=lambda x: x.withColumn("kg_source", F.lit("ec_medical_team")),
                inputs=["ingestion.raw.ec_medical_team.nodes@spark"],
                outputs="ingestion.int.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=["ec_medical_team"],
            ),
            node(
                func=lambda x: x.withColumn("kg_source", F.lit("ec_medical_team")),
                inputs=["ingestion.raw.ec_medical_team.edges@spark"],
                outputs="ingestion.int.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=["ec_medical_team"],
            ),
        ]
    )





