from kedro.pipeline import Pipeline, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create doc release pipeline."""
    # example name/release pipeline, all names can be changed
    return pipeline(
        [
            node(
                func=nodes.get_pks_from_nodes,
                inputs=["data_release.prm.kgx_edges", "data_release.prm.kgx_nodes"],
                outputs="document_kg.prm.pks",
                name="get_pks_from_kg",
                tags=["document_kg"],
            )
        ]
    )
