"""Ingestion pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

import pyspark.sql.functions as F
from .nodes import ingest_robokop_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # FUTURE: Use dynamic pipeline for this good first issue
    return pipeline(
        [
            # rtx-kg2
            node(
                func=lambda x: x.withColumn("kg_source", F.lit("rtx_kg2")),
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.int.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            node(
                func=lambda x: x.withColumn("kg_source", F.lit("rtx_kg2")),
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
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
            # robokop
            node(
                func=ingest_robokop_nodes,
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.int.robokop.nodes",
                name="write_robokop_nodes",
                tags=["robokop"],
            ),
            node(
                # FUTURE: Update selection
                func=lambda x: x.withColumn("kg_source", F.lit("robokop")),
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.int.robokop.edges",
                name="write_robokop_edges",
                tags=["robokop"],
            ),
        ]
    )
