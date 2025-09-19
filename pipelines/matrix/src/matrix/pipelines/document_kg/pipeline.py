from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Generate documentation and metadata as part of the release process pipeline."""

    return pipeline(
        [
            node(
                func=nodes.create_pks_integrated_metadata,
                inputs=[
                    "document_kg.prm.infores", 
                    "document_kg.prm.reusabledata",
                    "document_kg.prm.kgregistry",
                    "document_kg.prm.matrix_curated_pks",
                    "document_kg.prm.matrix_reviews_pks",
                    "document_kg.prm.pks_integrated_kg_list",
                    "document_kg.prm.mapping_reusabledata_infores",
                    "document_kg.prm.mapping_kgregistry_infores",
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
            )
        ]
    )
