from kedro.pipeline import Pipeline, node, pipeline

from . import nodes
from .tagging import generate_tags


def create_pipeline(**kwargs) -> Pipeline:
    """Create preprocessing pipeline."""
    return pipeline(
        [
            # -------------------------------------------------------------------------
            # EC Medical Team ingestion and enrichment
            # -------------------------------------------------------------------------
            node(
                func=nodes.process_medical_nodes,
                inputs=["preprocessing.raw.nodes", "params:preprocessing.name_resolution.url"],
                outputs="preprocessing.int.nodes",
                name="normalize_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.process_medical_edges,
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.raw.edges",
                ],
                outputs="preprocessing.int.edges",
                name="create_int_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
            node(
                func=lambda x, y: [x, y],
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.int.edges",
                ],
                outputs=["ingestion.raw.ec_medical_team.nodes@pandas", "ingestion.raw.ec_medical_team.edges@pandas"],
                name="produce_medical_kg",
                tags=["ec-medical-kg"],
            ),
            # -------------------------------------------------------------------------
            # EC Clinical Trials ingestion and enrichment
            # -------------------------------------------------------------------------
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
            # Drug List ingestion
            # -------------------------------------------------------------------------
            node(
                func=lambda x: x,
                inputs=["preprocessing.raw.drug_list"],
                outputs="ingestion.raw.drug_list.nodes@pandas",
                name="write_drug_list",
                tags=["drug-list"],
            ),
            # -------------------------------------------------------------------------
            # Disease List ingestion and enrichment
            # -------------------------------------------------------------------------
            node(
                func=generate_tags,
                inputs=[
                    "preprocessing.raw.disease_list",
                    "params:preprocessing.enrichment.model",
                    "params:preprocessing.enrichment.tags",
                ],
                outputs="ingestion.raw.disease_list.nodes@pandas",
                name="enrich_disease_list",
                tags=["disease-list"],
            ),
            # -------------------------------------------------------------------------
            # EveryCure GT  ingestion and preprocessing
            # -------------------------------------------------------------------------
            node(
                func=nodes.create_nodes_edges,
                inputs=[
                    "preprocessing.int.ec_ground_truth.combined",
                    "params:preprocessing.ground_truth_ec.subject",
                    "params:preprocessing.ground_truth_ec.object",
                ],
                outputs=[
                    "preprocessing.int.ec_ground_truth.nodes@pandas",
                    "preprocessing.int.ec_ground_truth.edges@pandas",
                ],
                name="create_ec_ground_truth_nodes_and_edges",
                tags=["ground-truth-ec"],
            ),
            node(
                func=lambda x, y: [x, y],
                inputs=[
                    "preprocessing.int.ec_ground_truth.nodes@pandas",
                    "preprocessing.int.ec_ground_truth.edges@pandas",
                ],
                outputs=[
                    "ingestion.raw.ec_ground_truth.nodes@pandas",
                    "ingestion.raw.ec_ground_truth.edges@pandas",
                ],
                name="produce_ec_ground_truth_kg",
                tags=["ground-truth-ec"],
            ),
            # -------------------------------------------------------------------------
            # KGML-xDTD GT ingestion and preprocessing
            # -------------------------------------------------------------------------
            node(
                func=nodes.create_gt,
                inputs={
                    "pos_df": "preprocessing.raw.kgml_xdtd_ground_truth.positives",
                    "neg_df": "preprocessing.raw.kgml_xdtd_ground_truth.negatives",
                },
                outputs="preprocessing.int.kgml_xdtd_ground_truth.combined",
                name="create_kgml_xdt_gt_dataframe",
                tags=["ground-truth-kgml"],
            ),
            node(
                func=nodes.create_nodes_edges,
                inputs=[
                    "preprocessing.int.kgml_xdtd_ground_truth.combined",
                    "params:preprocessing.ground_truth_kgml_xdtd.subject",
                    "params:preprocessing.ground_truth_kgml_xdtd.object",
                ],
                outputs=[
                    "preprocessing.int.kgml_xdtd_ground_truth.nodes@pandas",
                    "preprocessing.int.kgml_xdtd_ground_truth.edges@pandas",
                ],
                name="create_kgml_xdtd_ground_truth_nodes_and_edges",
                tags=["ground-truth-kgml"],
            ),
            node(
                func=lambda x, y: [x, y],
                inputs=[
                    "preprocessing.int.kgml_xdtd_ground_truth.nodes@pandas",
                    "preprocessing.int.kgml_xdtd_ground_truth.edges@pandas",
                ],
                outputs=[
                    "ingestion.raw.kgml_xdtd_ground_truth.nodes@pandas",
                    "ingestion.raw.kgml_xdtd_ground_truth.edges@pandas",
                ],
                name="produce_kgml_xdtd_ground_truth_kg",
                tags=["ground-truth-kgml"],
            ),
            # -------------------------------------------------------------------------
            # Feedback Loop
            # -------------------------------------------------------------------------
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.feedback_known_entities",
                outputs="preprocessing.int.feedback_known_entities",
                name="ingest_known_entities",
                tags=["feedback-loop"],
            ),
            node(
                func=nodes.create_nodes_edges,
                inputs=[
                    "preprocessing.int.feedback_known_entities",
                    "params:preprocessing.feedback_known_entities.subject",
                    "params:preprocessing.feedback_known_entities.object",
                ],
                outputs=[
                    "ingestion.raw.feedback_known_entities.nodes@pandas",
                    "ingestion.raw.feedback_known_entities.edges@pandas",
                ],
                name="produce_known_entities_datasets",
                tags=["feedback-loop"],
            ),
        ]
    )
