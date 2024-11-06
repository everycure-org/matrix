import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, pipeline
from matrix.kedro_extension import kubernetes_node


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # FUTURE: Use dynamic pipeline for this good first issue
    return pipeline(
        [
            # rtx-kg2
            kubernetes_node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.int.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            kubernetes_node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
            ),
            kubernetes_node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.curie_to_pmids@spark"],
                outputs="ingestion.int.rtx_kg2.curie_to_pmids",
                name="write_rtx_kg2_curie_to_pmids",
                tags=["rtx_kg2"],
            ),
            # ec-medical-team
            kubernetes_node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.nodes@spark"],
                outputs="ingestion.int.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=["ec_medical_team"],
            ),
            kubernetes_node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.edges@spark"],
                outputs="ingestion.int.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=["ec_medical_team"],
            ),
            # robokop
            kubernetes_node(
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.int.robokop.nodes",
                name="ingest_robokop_nodes",
                tags=["robokop"],
            ),
            kubernetes_node(
                # FUTURE: Update selection
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.int.robokop.edges",
                name="ingest_robokop_edges",
                tags=["robokop"],
            ),
        ]
    )
