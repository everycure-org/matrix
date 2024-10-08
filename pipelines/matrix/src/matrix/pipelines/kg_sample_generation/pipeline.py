import pandas as pd


from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """Create KG sample pipeline."""
    return pipeline(
        [
            # Takes a sample of RTX
            node(
                func=nodes.get_random_selection_from_rtx,
                inputs={
                    "nodes": "ingestion.raw.rtx_kg2.nodes@spark",
                    "edges": "ingestion.raw.rtx_kg2.edges@spark",
                },
                outputs={
                    "nodes": "ingestion.int.rtx_kg2.nodes",
                    "edges": "ingestion.int.rtx_kg2.edges",
                },
                name="sampled_kg2_datasets",
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


