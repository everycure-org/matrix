from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Generate documentation and metadata as part of the release process pipeline."""

    return pipeline(
        [
            node(
                func=nodes.extract_pks_from_unified_edges,
                inputs="document_kg.raw.unified_edges@spark",
                outputs="document_kg.prm.pks_integrated_kg_list",
                name="extract_pks_from_unified_edges",
                tags=["document_kg"],
            ),
            node(
                func=nodes.create_pks_integrated_metadata,
                inputs=[
                    "document_kg.raw.infores",
                    "document_kg.raw.reusabledata",
                    "document_kg.raw.kgregistry",
                    "document_kg.raw.matrix_curated_pks@pandas",
                    "document_kg.raw.matrix_reviews_pks@pandas",
                    "document_kg.int.pks_integrated_kg_list",
                    "document_kg.raw.mapping_reusabledata_infores",
                    "document_kg.raw.mapping_kgregistry_infores",
                ],
                outputs=[
                    "document_kg.prm.pks_yaml",
                ],
                name="create_pks_integrated_metadata",
                tags=["document_kg"],
            ),
            node(
                func=nodes.create_pks_documentation,
                inputs=[
                    "document_kg.prm.pks_yaml",
                ],
                outputs=[
                    "document_kg.prm.pks_md",
                ],
                name="create_pks_documentation",
                tags=["document_kg"],
            ),
        ]
    )
