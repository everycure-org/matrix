from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

# NOTE: Preprocessing pipeline is not well optimized and thus might take a while to run.


def create_embiology_pipeline() -> Pipeline:
    """Embiology cleaning and preprocessing"""
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.attr",
                outputs="preprocessing.int.embiology.attr@pandas",
                name="write_embiology_attr",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.ref_pub",
                outputs="preprocessing.int.embiology.ref_pub@pandas",
                name="write_embiology_ref_pub",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.nodes",
                outputs="preprocessing.int.embiology.nodes@pandas",
                name="write_embiology_nodes",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.edges",
                outputs="preprocessing.int.embiology.edges@pandas",
                name="write_embiology_edges",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.manual_id_mapping",
                outputs="preprocessing.int.embiology.manual_id_mapping@pandas",
                name="write_embiology_id_mapping",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.manual_name_mapping",
                outputs="preprocessing.int.embiology.manual_name_mapping@pandas",
                name="write_embiology_name_mapping",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=nodes.prepare_normalized_identifiers,
                inputs=[
                    "preprocessing.int.embiology.attr@spark",
                    "preprocessing.int.embiology.nodes@spark",
                    "preprocessing.int.embiology.manual_id_mapping@spark",
                    "preprocessing.int.embiology.manual_name_mapping@spark",
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
            node(
                func=nodes.add_edge_attributes,
                inputs=[
                    "preprocessing.int.embiology.ref_pub@spark",
                ],
                outputs="preprocessing.int.embiology.attributes",
                name="prepare_edges_attributes",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.prepare_edges,
                inputs=[
                    "preprocessing.int.embiology.edges@spark",
                    "preprocessing.int.embiology.attributes",
                    "params:preprocessing.embiology.edges.biolink_mapping",
                ],
                outputs="preprocessing.prm.embiology.edges",
                name="prepare_edges",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.deduplicate_and_clean,
                inputs=[
                    "preprocessing.prm.embiology.nodes",
                    "preprocessing.prm.embiology.edges",
                ],
                outputs=[
                    "preprocessing.prm.embiology.nodes_final",
                    "preprocessing.prm.embiology.edges_final",
                ],
                name="final_clean_embiology_kg",
                tags=["embiology-kg"],
            ),
        ]
    )


def create_ec_clinical_data_pipeline() -> Pipeline:
    """EC Clinical Data ingestion and name->id mapping"""
    return pipeline(
        [
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
        ]
    )


def create_ec_medical_team_pipeline() -> Pipeline:
    """EC Medical Team ingestion and name-> id mapping"""
    return pipeline(
        [
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


def create_pipeline() -> Pipeline:
    """Create preprocessing pipeline."""
    return pipeline(
        create_embiology_pipeline(),
        create_ec_clinical_data_pipeline(),
        create_ec_medical_team_pipeline(),
    )
