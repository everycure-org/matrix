from kedro.pipeline import Pipeline, node, pipeline

from . import nodes
from .tagging import generate_tags


def create_pipeline(**kwargs) -> Pipeline:
    """Create preprocessing pipeline."""
    return pipeline(
        [
            node(
                func=nodes.add_source_and_target_to_clinical_trails,
                inputs={
                    "df": "preprocessing.raw.clinical_trials_data",
                    "resolver_url": "params:preprocessing.name_resolution.url",
                },
                outputs="preprocessing.int.mapped_clinical_trials_data",
                name="mapped_clinical_trials_data",
                tags=["ec-clinical-trials-data"],
            ),
            # NOTE: Clean up the clinical trial data and write it to the GCS bucket
            node(
                func=nodes.clean_clinical_trial_data,
                inputs=[
                    "preprocessing.int.mapped_clinical_trials_data",
                ],
                outputs={
                    "nodes": "ingestion.raw.ec_clinical_trails.nodes@pandas",
                    "edges": "ingestion.raw.ec_clinical_trails.edges@pandas",
                },
                name="clean_clinical_trial_data",
                tags=["ec-clinical-trials-data"],
            ),
            # -------------------------------------------------------------------------
            # EC Medical Team ingestion and enrichment
            # -------------------------------------------------------------------------
            node(
                func=nodes.process_medical_nodes,
                inputs=["preprocessing.raw.nodes", "params:preprocessing.name_resolution.url"],
                outputs="ingestion.raw.ec_medical_team.nodes@pandas",
                name="normalize_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.process_medical_edges,
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.raw.edges",
                ],
                outputs="ingestion.raw.ec_medical_team.edges@pandas",
                name="create_int_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
        ]
    )
