import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, node, pipeline

from matrix.tags import NodeTags


def create_pipeline(**kwargs) -> Pipeline:
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
                tags=[NodeTags.RTX_KG2.value],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=[NodeTags.RTX_KG2.value],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.curie_to_pmids@spark"],
                outputs="ingestion.int.rtx_kg2.curie_to_pmids",
                name="write_rtx_kg2_curie_to_pmids",
                tags=[NodeTags.RTX_KG2.value],
            ),
            # ec-medical-team
            node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.nodes@spark"],
                outputs="ingestion.int.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=[NodeTags.EC_MEDICAL_TEAM.value],
            ),
            node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.edges@spark"],
                outputs="ingestion.int.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=[NodeTags.EC_MEDICAL_TEAM.value],
            ),
            # robokop
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.int.robokop.nodes",
                name="ingest_robokop_nodes",
                tags=[NodeTags.ROBOKOP.value],
            ),
            node(
                # FUTURE: Update selection
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.int.robokop.edges",
                name="ingest_robokop_edges",
                tags=[NodeTags.ROBOKOP.value],
            ),
        ]
    )
