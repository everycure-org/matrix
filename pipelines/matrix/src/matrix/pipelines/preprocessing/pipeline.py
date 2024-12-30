from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import ArgoNode

from . import nodes


# NOTE: This pipeline in highly preliminary and used for ingestion of the
# medical data provided in Google Sheets __ONLY__.
def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Normalize nodes
            ArgoNode(
                func=nodes.create_int_nodes,
                inputs={
                    "nodes": "preprocessing.raw.nodes",
                    "name_resolver": "params:preprocessing.translator.name_resolver",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs="preprocessing.int.nodes",
                name="normalize_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            ArgoNode(
                func=nodes.create_int_edges,
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.raw.edges",
                ],
                outputs="preprocessing.int.edges",
                name="create_int_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
            ArgoNode(
                func=nodes.create_prm_edges,
                inputs=[
                    "preprocessing.int.edges",
                ],
                outputs="ingestion.raw.ec_medical_team.edges@pandas",
                name="create_prm_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
            ArgoNode(
                func=nodes.create_prm_nodes,
                inputs=[
                    "preprocessing.int.nodes",
                ],
                outputs="ingestion.raw.ec_medical_team.nodes@pandas",
                name="create_prm_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            # NOTE: Take raw clinical trial data and map the "name" to "curie" using the synonymizer
            ArgoNode(
                func=nodes.map_name_to_curie,
                inputs={
                    "df": "preprocessing.raw.clinical_trials_data",
                    "name_resolver": "params:preprocessing.translator.name_resolver",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "drug_types": "params:modelling.drug_types",
                    "disease_types": "params:modelling.disease_types",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs="preprocessing.int.mapped_clinical_trials_data",
                name="mapped_clinical_trials_data",
                tags=["ec-clinical-trials-data"],
            ),
            # NOTE: Clean up the clinical trial data and write it to the GCS bucket
            ArgoNode(
                func=nodes.clean_clinical_trial_data,
                inputs=[
                    "preprocessing.int.mapped_clinical_trials_data",
                ],
                outputs="ingestion.raw.clinical_trials_data",
                name="clean_clinical_trial_data",
                tags=["ec-clinical-trials-data"],
            ),
            ArgoNode(
                func=nodes.clean_drug_list,
                inputs={
                    "drug_df": "preprocessing.raw.drug_list",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs="ingestion.raw.drug_list@pandas",
                name="resolve_drug_list",
                tags=["drug-list"],
            ),
            ArgoNode(
                inputs="ingestion.raw.drug_list@pandas",
                outputs="ingestion.reporting.drug_list",
                name="write_drug_list_to_gsheets",
            ),
            # FUTURE: Remove this node once we have a new disease list with tags
            ArgoNode(
                func=nodes.enrich_disease_list,
                inputs=[
                    "preprocessing.raw.disease_list",
                    "params:preprocessing.enrichment_tags",
                ],
                outputs="preprocessing.raw.enriched_disease_list",
                name="enrich_disease_list",
                tags=["disease-list"],
            ),
            ArgoNode(
                func=nodes.clean_disease_list,
                inputs={
                    "disease_df": "preprocessing.raw.enriched_disease_list",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs="ingestion.raw.disease_list@pandas",
                name="resolve_disease_list",
                tags=["disease-list"],
            ),
            ArgoNode(
                inputs="ingestion.raw.disease_list@pandas",
                outputs="ingestion.reporting.disease_list",
                name="write_disease_list_to_gsheets",
            ),
            ArgoNode(
                func=nodes.clean_input_sheet,
                inputs={
                    "input_df": "preprocessing.raw.infer_sheet",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs="inference.raw.normalized_inputs",
                name="clean_input_sheet",
                tags=["inference-input"],
            ),
            ArgoNode(
                func=nodes.clean_gt_data,
                inputs={
                    "pos_df": "preprocessing.raw.ground_truth.positives",
                    "neg_df": "preprocessing.raw.ground_truth.negatives",
                    "endpoint": "params:preprocessing.translator.normalizer",
                    "conflate": "params:integration.nodenorm.conflate",
                    "drug_chemical_conflate": "params:integration.nodenorm.drug_chemical_conflate",
                    "batch_size": "params:integration.nodenorm.batch_size",
                    "parallelism": "params:integration.nodenorm.parallelism",
                },
                outputs=["modelling.raw.ground_truth.positives@pandas", "modelling.raw.ground_truth.negatives@pandas"],
                name="resolve_gt",
                tags=["ground-truth"],
            ),
        ]
    )
