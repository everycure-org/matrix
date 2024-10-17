"""Ingestion pipeline."""

import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, node, pipeline


def create_ingestion_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # FUTURE: Use dynamic pipeline for this good first issue
    return pipeline(
        [
            # rtx-kg2
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.int.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
            ),
            # ec-medical-team
            node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.nodes@spark"],
                outputs="ingestion.int.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=["ec_medical_team"],
            ),
            node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.edges@spark"],
                outputs="ingestion.int.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=["ec_medical_team"],
            ),
            # robokop
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.int.robokop.nodes",
                name="ingest_robokop_nodes",
                tags=["robokop"],
            ),
            node(
                # FUTURE: Update selection
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.int.robokop.edges",
                name="ingest_robokop_edges",
                tags=["robokop"],
            ),
        ]
    )
