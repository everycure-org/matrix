from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


# NOTE: Preprocessing pipeline is not well optimized and thus might take a while to run.
def create_pipeline(**kwargs) -> Pipeline:
    """Create preprocessing pipeline."""
    return pipeline(
        [
            # -------------------------------------------------------------------------
            # Embiology cleaning and preprocessing
            # -------------------------------------------------------------------------
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.attr",
                outputs="preprocessing.int.embiology.attr@pandas",
                name="write_embiology_attr",
                tags=["embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.nodes",
                outputs="preprocessing.int.embiology.nodes@pandas",
                name="write_embiology_nodes",
                tags=["embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.edges",
                outputs="preprocessing.int.embiology.edges@pandas",
                name="write_embiology_edges",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.prepare_normalized_identifiers,
                inputs=[
                    "preprocessing.int.embiology.attr@spark",
                    "params:preprocessing.embiology.attr.identifiers_mapping",
                    "params:preprocessing.embiology.normalization",
                ],
                outputs="preprocessing.int.embiology.identifiers@pandas",
                name="prepare_normalized_identifiers",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.prepare_nodes,
                inputs=[
                    "preprocessing.int.embiology.nodes@spark",
                    "preprocessing.int.embiology.identifiers@spark",
                    "params:preprocessing.embiology.nodes.biolink_mapping",
                ],
                outputs="preprocessing.prm.embiology.nodes",
                name="prepare_nodes",
                tags=["embiology-kg"],
            ),
            # node(
            #     func=nodes.deduplicate_and_clean,
            #     inputs=[
            #         "preprocessing.int.embiology.nodes",
            #         "preprocessing.int.embiology.edges",
            #     ],
            #     outputs=[
            #         "preprocessing.prm.embiology.nodes",
            #         "preprocessing.prm.embiology.edges",
            #     ],
            #     name="final_clean_embiology_kg",
            #     tags=["embiology-kg"],
            # ),
            # -------------------------------------------------------------------------
            # EC Clinical Data ingestion and name->id mapping
            # -------------------------------------------------------------------------
            node(
                func=nodes.add_source_and_target_to_clinical_trails,
                inputs={
                    "df": "preprocessing.raw.clinical_trials_data",
                    "resolver_url": "params:preprocessing.name_resolution.url",
                    "batch_size": "params:preprocessing.name_resolution.batch_size",
                },
                outputs="preprocessing.int.mapped_clinical_trials_data",
                name="mapped_clinical_trials_data",
                tags=["ec-clinical-trials-data"],
            ),
            node(
                func=nodes.clean_clinical_trial_data,
                inputs={"df": "preprocessing.int.mapped_clinical_trials_data"},
                outputs={
                    "nodes": "ingestion.raw.ec_clinical_trails.nodes@pandas",
                    "edges": "ingestion.raw.ec_clinical_trails.edges@pandas",
                },
                name="clean_clinical_trial_data",
                tags=["ec-clinical-trials-data"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.int.mapped_clinical_trials_data",
                outputs="preprocessing.reporting.mapped_clinical_trials_data",
                name="report_clinical_trial_data",
                tags=["ec-clinical-trials-data"],
            ),
            # -------------------------------------------------------------------------
            # EC Medical Team ingestion and name-> id mapping
            # -------------------------------------------------------------------------
            node(
                func=nodes.process_medical_nodes,
                inputs={
                    "df": "preprocessing.raw.ec_medical_team.nodes",
                    "resolver_url": "params:preprocessing.name_resolution.url",
                    "batch_size": "params:preprocessing.name_resolution.batch_size",
                },
                outputs="ingestion.raw.ec_medical_team.nodes@pandas",
                name="normalize_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.process_medical_edges,
                inputs={
                    "int_nodes": "ingestion.raw.ec_medical_team.nodes@pandas",
                    "raw_edges": "preprocessing.raw.ec_medical_team.edges",
                },
                outputs="ingestion.raw.ec_medical_team.edges@pandas",
                name="create_int_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.report_to_gsheets,
                inputs=[
                    "ingestion.raw.ec_medical_team.nodes@pandas",
                    "preprocessing.raw.ec_medical_team.nodes",
                    "params:preprocessing.medical_kg_gsheets.nodes",
                ],
                outputs="preprocessing.reporting.ec_medical_team.nodes",
                name="report_ec_medical_team_nodes",
                tags=["ec-medical-kg"],
            ),
            node(
                func=nodes.report_to_gsheets,
                inputs=[
                    "ingestion.raw.ec_medical_team.edges@pandas",
                    "preprocessing.raw.ec_medical_team.edges",
                    "params:preprocessing.medical_kg_gsheets.edges",
                ],
                outputs="preprocessing.reporting.ec_medical_team.edges",
                name="report_ec_medical_team_edges",
                tags=["ec-medical-kg"],
            ),
        ]
    )
