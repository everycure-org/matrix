import pyspark.sql.functions as F
from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import (
    KUBERNETES_DEFAULT_REQUEST_CPU,
    KUBERNETES_DEFAULT_LIMIT_CPU,
    KUBERNETES_DEFAULT_REQUEST_RAM,
    KUBERNETES_DEFAULT_LIMIT_RAM,
    ArgoNodeConfig,
    argo_node,
)


ingestion_argo_node_config = ArgoNodeConfig(
    cpu_request=KUBERNETES_DEFAULT_REQUEST_CPU,
    cpu_limit=KUBERNETES_DEFAULT_LIMIT_CPU,
    memory_request=KUBERNETES_DEFAULT_REQUEST_RAM,
    memory_limit=KUBERNETES_DEFAULT_LIMIT_RAM,
    use_gpu=False,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # FUTURE: Use dynamic pipeline for this good first issue
    return pipeline(
        [
            # rtx-kg2
            argo_node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.int.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
                argo_config=ingestion_argo_node_config,
            ),
            argo_node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
                argo_config=ingestion_argo_node_config,
            ),
            argo_node(
                func=lambda x: x,
                inputs=["ingestion.raw.rtx_kg2.curie_to_pmids@spark"],
                outputs="ingestion.int.rtx_kg2.curie_to_pmids",
                name="write_rtx_kg2_curie_to_pmids",
                tags=["rtx_kg2"],
                argo_config=ingestion_argo_node_config,
            ),
            # ec-medical-team
            argo_node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.nodes@spark"],
                outputs="ingestion.int.ec_medical_team.nodes",
                name="write_ec_medical_team_nodes",
                tags=["ec_medical_team"],
                argo_config=ingestion_argo_node_config,
            ),
            argo_node(
                func=lambda x: x.withColumn("upstream_data_source", F.array(F.lit("ec_medical_team"))),
                inputs=["ingestion.raw.ec_medical_team.edges@spark"],
                outputs="ingestion.int.ec_medical_team.edges",
                name="write_ec_medical_team_edges",
                tags=["ec_medical_team"],
                argo_config=ingestion_argo_node_config,
            ),
            # robokop
            argo_node(
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.nodes@spark"],
                outputs="ingestion.int.robokop.nodes",
                name="ingest_robokop_nodes",
                tags=["robokop"],
                argo_config=ingestion_argo_node_config,
            ),
            argo_node(
                # FUTURE: Update selection
                func=lambda x: x,
                inputs=["ingestion.raw.robokop.edges@spark"],
                outputs="ingestion.int.robokop.edges",
                name="ingest_robokop_edges",
                tags=["robokop"],
                argo_config=ingestion_argo_node_config,
            ),
        ]
    )
