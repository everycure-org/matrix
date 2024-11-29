from kedro.pipeline import Pipeline, pipeline, node

from matrix.pipelines.embeddings.nodes import ingest_edges, ingest_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
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
            # NOTE: Enable if you want embeddings
            # node(
            #     func=lambda _, x: x,
            #     inputs=["data_release.prm.kg_nodes", "embeddings.feat.nodes"],
            #     outputs="data_release.prm.kg_embeddings",
            # )
            # need nodes that bring the nodes/edges to BigQuery
        ]
    )
