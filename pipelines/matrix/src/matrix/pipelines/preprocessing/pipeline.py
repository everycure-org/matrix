"""Preprocessing pipeline."""
from functools import partial
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


# NOTE: This pipeline in highly preliminary and used for ingestion of the
# medical data provided in Google Sheets __ONLY__.
def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Normalize nodes
            node(
                func=nodes.create_int_nodes,
                inputs=[
                    "preprocessing.raw.nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.nodes",
                name="normalize_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.create_int_edges,
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.raw.edges",
                ],
                outputs="preprocessing.int.edges",
                name="create_int_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.create_prm_edges,
                inputs=[
                    "preprocessing.int.edges",
                ],
                outputs="ingestion.raw.ec_medical_team.edges@pandas",
                name="create_prm_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.create_prm_nodes,
                inputs=[
                    "preprocessing.int.nodes",
                ],
                outputs="ingestion.raw.ec_medical_team.nodes@pandas",
                name="create_prm_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            # NOTE: Take raw clinical trial data and map the "name" to "curie" using the synonymizer
            node(
                func=nodes.map_name_to_curie,
                inputs=[
                    "preprocessing.raw.clinical_trials_data",
                    "params:preprocessing.synonymizer_endpoint",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
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
                outputs="ingestion.raw.clinical_trials_data",
                name="clean_clinical_trial_data",
                tags=["ec-clinical-trials-data"],
            ),
            node(
                func=nodes.clean_drug_list,
                inputs=[
                    "preprocessing.raw.drug_list",
                    "params:preprocessing.synonymizer_endpoint",
                    "params:modelling.drug_types",
                ],
                outputs="ingestion.raw.drug_list",
                name="resolve_drug_list",
                tags=["drug-list"],
            ),
            node(
                func=nodes.clean_disease_list,
                inputs=[
                    "preprocessing.raw.disease_list",
                    "params:preprocessing.synonymizer_endpoint",
                    "params:modelling.disease_types",
                ],
                outputs="ingestion.raw.disease_list",
                name="resolve_disease_list",
                tags=["disease-list"],
            ),
        ]
    )
