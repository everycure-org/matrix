from kedro.pipeline import Pipeline, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create doc release pipeline."""
    # example name/release pipeline, all names can be changed
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
                    "document_kg.prm.pks",
                    "document_kg.prm.pks_md"
                ],
                name="create_pks_integrated_metadata",
                tags=["document_kg"],
            )
        ]
    )
