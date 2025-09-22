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
                name="primekg_build_nodes",
                tags=["primekg"],
            ),
            node(
                func=nodes.primekg_build_edges,
                inputs="preprocessing.raw.primekg.kg@polars",
                outputs="preprocessing.int.primekg.edges",
                name="primekg_build_edges",
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
                    "params:preprocessing.normalization",
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


def create_pipeline() -> Pipeline:
    """Create preprocessing pipeline."""
    return pipeline(
        [
            create_primekg_pipeline(),
            create_embiology_pipeline(),
        ]
    )
