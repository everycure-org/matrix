import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, pipeline, node
from matrix.kedro4argo_node import ArgoNode


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # FUTURE: Use dynamic pipeline for this good first issue
    return pipeline(
        [
            # rtx-kg2
            ArgoNode(
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.int.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            ArgoNode(
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
            ),
            ArgoNode(
                inputs=["ingestion.raw.rtx_kg2.curie_to_pmids@spark"],
                outputs="ingestion.int.rtx_kg2.curie_to_pmids",
                name="write_rtx_kg2_curie_to_pmids",
                tags=["rtx_kg2"],
            ),
            # ec-medical-team
            ArgoNode(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.nodes@spark"],
                outputs="ingestion.int.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=["ec_medical_team"],
            ),
            ArgoNode(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.edges@spark"],
                outputs="ingestion.int.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=["ec_medical_team"],
            ),
            # robokop
            ArgoNode(
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.int.robokop.nodes",
                name="ingest_robokop_nodes",
                tags=["robokop"],
            ),
            ArgoNode(
                # FUTURE: Update selection
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.int.robokop.edges",
                name="ingest_robokop_edges",
                tags=["robokop"],
            ),
            # spoke
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.spoke.nodes@spark"],
                outputs="ingestion.int.spoke.nodes",
                name="ingest_spoke_nodes",
                tags=["spoke"],
            ),
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.spoke.edges@spark"],
                outputs="ingestion.int.spoke.edges",
                name="ingest_spoke_edges",
                tags=["spoke"],
            ),
        ]
    )
