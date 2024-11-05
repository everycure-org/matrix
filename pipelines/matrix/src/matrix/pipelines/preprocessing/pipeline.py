from kedro.pipeline import Pipeline, node, pipeline

from . import nodes
from matrix.tags import NodeTags


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
                tags=[NodeTags.EC_MEDICAL_KG.value],
            ),
            node(
                func=nodes.create_int_edges,
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.raw.edges",
                ],
                outputs="preprocessing.int.edges",
                name="create_int_ec_medical_team_edges",
                tags=[NodeTags.EC_MEDICAL_KG.value],
            ),
            node(
                func=nodes.create_prm_edges,
                inputs=[
                    "preprocessing.int.edges",
                ],
                outputs="ingestion.raw.ec_medical_team.edges@pandas",
                name="create_prm_ec_medical_team_edges",
                tags=[NodeTags.EC_MEDICAL_KG.value],
            ),
            node(
                func=nodes.create_prm_nodes,
                inputs=[
                    "preprocessing.int.nodes",
                ],
                outputs="ingestion.raw.ec_medical_team.nodes@pandas",
                name="create_prm_ec_medical_team_nodes",
                tags=[NodeTags.EC_MEDICAL_KG.value],
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
                tags=[NodeTags.EC_CLINICAL_TRIALS_DATA.value],
            ),
            # NOTE: Clean up the clinical trial data and write it to the GCS bucket
            node(
                func=nodes.clean_clinical_trial_data,
                inputs=[
                    "preprocessing.int.mapped_clinical_trials_data",
                ],
                outputs="ingestion.raw.clinical_trials_data",
                name="clean_clinical_trial_data",
                tags=[NodeTags.EC_CLINICAL_TRIALS_DATA.value],
            ),
            node(
                func=nodes.clean_drug_list,
                inputs=[
                    "preprocessing.raw.drug_list",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="ingestion.raw.drug_list@pandas",
                name="resolve_drug_list",
                tags=[NodeTags.DRUG_LIST.value],
            ),
            node(
                func=lambda x: x,
                inputs="ingestion.raw.drug_list@pandas",
                outputs="ingestion.reporting.drug_list",
                name="write_drug_list_to_gsheets",
            ),
            # FUTURE: Remove this node once we have a new disease list with tags
            node(
                func=nodes.enrich_disease_list,
                inputs=["preprocessing.raw.disease_list", "params:preprocessing.enrichment_tags"],
                outputs="preprocessing.raw.enriched_disease_list",
                name="enrich_disease_list",
                tags=[NodeTags.DISEASE_LIST.value],
            ),
            node(
                func=nodes.clean_disease_list,
                inputs=[
                    "preprocessing.raw.enriched_disease_list",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="ingestion.raw.disease_list@pandas",
                name="resolve_disease_list",
                tags=[NodeTags.DISEASE_LIST.value],
            ),
            node(
                func=lambda x: x,
                inputs="ingestion.raw.disease_list@pandas",
                outputs="ingestion.reporting.disease_list",
                name="write_disease_list_to_gsheets",
            ),
            node(
                func=nodes.clean_input_sheet,
                inputs=[
                    "preprocessing.raw.infer_sheet",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="inference.raw.normalized_inputs",
                name="clean_input_sheet",
                tags=[NodeTags.INFERENCE_INPUT.value],
            ),
        ]
    )
