"""Pipeline to release data."""
from kedro.pipeline import Pipeline, node, pipeline

from matrix.pipelines.embeddings.nodes import ingest_nodes, ingest_edges


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            node(
                func=ingest_nodes,
                inputs=["integration.prm.unified_nodes"],
                outputs="release.prm.kg_nodes",
                name="ingest_kg_nodes",
            ),
            node(
                func=ingest_edges,
                inputs=["release.prm.kg_nodes", "integration.prm.unified_edges"],
                outputs="release.prm.kg_edges",
                name="ingest_kg_edges",
            ),
            # NOTE: Enable if you want embeddings
            # node(
            #     func=lambda _, x: x,
            #     inputs=["release.prm.kg_nodes", "embeddings.feat.nodes"],
            #     outputs="release.prm.kg_embeddings",
            # )
        ]
    )
