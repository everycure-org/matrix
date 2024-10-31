"""Pipeline to release data."""

from kedro.pipeline import Pipeline, node

from matrix.pipelines.data_release.pipeline import create_pipeline as create_data_release_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Create data release pipeline, with embeddings.

    This is an extension of the data release pipeline, adding the embeddings.
    """
    data_release_pipeline = create_data_release_pipeline()
    data_release_pipeline += node(
        func=lambda _, x: x,
        inputs=["data_release.prm.kg_nodes", "embeddings.feat.nodes"],
        outputs="data_release.prm.kg_embeddings",
    )

    return data_release_pipeline
