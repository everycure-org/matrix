from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

# NOTE: Preprocessing pipeline is not well optimized and thus might take a while to run.


def create_primekg_pipeline() -> Pipeline:
    """PrimeKG preprocessing"""
    return pipeline(
        [
            node(
                func=nodes.primekg_build_nodes,
                inputs={
                    "nodes": "preprocessing.raw.primekg.nodes@polars",
                    "drug_features": "preprocessing.raw.primekg.drug_features@polars",
                    "disease_features": "preprocessing.raw.primekg.disease_features@polars",
                },
                outputs="preprocessing.int.primekg.nodes",
                name="build_primekg_nodes",
                tags=["primekg"],
            ),
            node(
                func=nodes.primekg_build_edges,
                inputs=[
                    "preprocessing.raw.primekg.kg@polars",
                ],
                outputs="preprocessing.int.primekg.edges",
                name="build_primekg_edges",
                tags=["primekg"],
            ),
        ]
    )


def create_embiology_pipeline() -> Pipeline:
    """Embiology cleaning and preprocessing"""
    return pipeline(
        [
            # Copying these data sources locally improves performance in the later steps.
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.node_attributes",
                outputs="preprocessing.int.embiology.node_attributes@pandas",
                name="write_embiology_attr_to_tmp",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.ref_pub",
                outputs="preprocessing.int.embiology.ref_pub@pandas",
                name="write_embiology_ref_pub_to_tmp",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.nodes",
                outputs="preprocessing.int.embiology.nodes@pandas",
                name="write_embiology_nodes_to_tmp",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.edges",
                outputs="preprocessing.int.embiology.edges@pandas",
                name="write_embiology_edges_to_tmp",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.manual_id_mapping",
                outputs="preprocessing.int.embiology.manual_id_mapping@pandas",
                name="write_embiology_id_mapping_to_tmp",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=lambda x: x,
                inputs="preprocessing.raw.embiology.manual_name_mapping",
                outputs="preprocessing.int.embiology.manual_name_mapping@pandas",
                name="write_embiology_name_mapping_to_tmp",
                tags=["ingest-embiology-kg"],
            ),
            node(
                func=nodes.get_embiology_node_attributes_normalised_ids,
                inputs=[
                    "preprocessing.int.embiology.node_attributes@spark",
                    "preprocessing.int.embiology.nodes@spark",
                    "preprocessing.int.embiology.manual_id_mapping@spark",
                    "preprocessing.int.embiology.manual_name_mapping@spark",
                    "params:preprocessing.embiology.attr.identifiers_mapping",
                    "params:preprocessing.embiology.normalization",
                ],
                outputs="preprocessing.int.embiology.node_attributes_normalised_ids@pandas",
                name="get_embiology_node_attributes_normalised_ids",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.normalise_embiology_nodes,
                inputs=[
                    "preprocessing.int.embiology.nodes@spark",
                    "preprocessing.int.embiology.node_attributes_normalised_ids@spark",
                    "params:preprocessing.embiology.nodes.biolink_mapping",
                ],
                outputs="preprocessing.prm.embiology.nodes",
                name="normalise_embiology_nodes",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.generate_embiology_edge_attributes,
                inputs=[
                    "preprocessing.int.embiology.ref_pub@spark",
                ],
                outputs="preprocessing.int.embiology.edge_attributes",
                name="generate_embiology_edge_attributes",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.prepare_embiology_edges,
                inputs=[
                    "preprocessing.int.embiology.edges@spark",
                    "preprocessing.int.embiology.edge_attributes",
                    "params:preprocessing.embiology.edges.biolink_mapping",
                ],
                outputs="preprocessing.prm.embiology.edges",
                name="prepare_embiology_edges",
                tags=["embiology-kg"],
            ),
            node(
                func=nodes.deduplicate_and_clean_embiology_kg,
                inputs=[
                    "preprocessing.prm.embiology.nodes",
                    "preprocessing.prm.embiology.edges",
                ],
                outputs=[
                    "preprocessing.prm.embiology.nodes_final",
                    "preprocessing.prm.embiology.edges_final",
                ],
                name="deduplicate_and_clean_embiology_kg",
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
        [
            create_primekg_pipeline(),
            create_embiology_pipeline(),
            create_ec_clinical_data_pipeline(),
            create_ec_medical_team_pipeline(),
        ]
    )
