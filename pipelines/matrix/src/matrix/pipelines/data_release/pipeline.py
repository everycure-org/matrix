from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ArgoNode
from matrix.pipelines.data_release.nodes import release_edges, release_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            ArgoNode(
                func=release_edges,
                inputs=["integration.prm.unified_edges"],
                outputs=["data_release.prm.kgx_edges", "data_release.hf.kgx_edges"],
                name="write_edges_to_kgx",
                tags=["kgx"],
            ),
            ArgoNode(
                func=release_nodes,
                inputs=["integration.prm.unified_nodes"],
                outputs=["data_release.prm.kgx_nodes", "data_release.hf.kgx_nodes"],
                name="write_nodes_to_kgx",
                tags=["kgx"],
            ),
            # NOTE: Enable when the embeddings pipeline worked prior to this pipeline
            # # release to neo4j
            # ArgoNode(
            #     func=lambda x: x,
            #     inputs=["embeddings.feat.nodes"],
            #     outputs="data_release.feat.nodes_with_embeddings",
            #     name="ingest_nodes_with_embeddings",
            # ),
            # NOTE: Enable if you want embeddings
            # ArgoNode(
            #     func=lambda _, x: x,
            #     inputs=["data_release.prm.kg_nodes", "embeddings.feat.nodes"],
            #     outputs="data_release.prm.kg_embeddings",
            # )
            # need nodes that bring the nodes/edges to BigQuery
        ]
    )
