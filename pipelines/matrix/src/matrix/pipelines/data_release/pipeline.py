from kedro.pipeline import Pipeline, pipeline, node

from matrix.pipelines.embeddings.nodes import ingest_edges, ingest_nodes
from matrix.pipelines.data_release.nodes import filtered_edges_to_kgx, filtered_nodes_to_kgx

def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs=["embeddings.feat.nodes"],
                outputs="data_release.feat.nodes_with_embeddings",
                name="ingest_nodes_with_embeddings",
            ),
            node(
                func=ingest_nodes,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.kg_nodes",
                name="ingest_kg_nodes",
                tags=["neo4j"],
            ),
            node(
                func=ingest_edges,
                inputs=["data_release.prm.kg_nodes", "integration.prm.filtered_edges"],
                outputs="data_release.prm.kg_edges",
                name="ingest_kg_edges",
                tags=["neo4j"],
            ),
            node(
                func=filtered_edges_to_kgx,
                inputs=["integration.prm.filtered_edges"],
                outputs="data_release.prm.kgx_edges",
                name="write_edges_to_kgx",
                tags=["kgx"],
            ),
            node(
                func=filtered_nodes_to_kgx,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.kgx_nodes",
                name="write_nodes_to_kgx",
                tags=["kgx"],
            ),
            # NOTE: Enable if you want embeddings
            # node(
            #     func=lambda _, x: x,
            #     inputs=["data_release.prm.kg_nodes", "embeddings.feat.nodes"],
            #     outputs="data_release.prm.kg_embeddings",
            # )
            # need nodes that bring the nodes/edges to BigQuery
        ]
    )
