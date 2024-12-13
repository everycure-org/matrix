from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import ArgoNode
from matrix.pipelines.embeddings.nodes import ingest_edges, ingest_nodes

# Last node is made explicit because there's a kedro hook after_node_run
# being triggered after the completion of the last node of this pipeline.
last_node = (
    ArgoNode(
        func=ingest_edges,
        inputs=["data_release.prm.kg_nodes", "integration.prm.filtered_edges"],
        outputs="data_release.prm.kg_edges",
        name="ingest_kg_edges",
        tags=["neo4j"],
    ),
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            # release to bigquery
            ArgoNode(
                func=lambda x: x,
                inputs=["integration.prm.filtered_edges"],
                outputs="data_release.prm.bigquery_edges",
                name="release_edges_to_bigquery",
            ),
            ArgoNode(
                func=lambda x: x,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.bigquery_nodes",
                name="release_nodes_to_bigquery",
            ),
            # release to neo4j
            ArgoNode(
                func=lambda x: x,
                inputs=["embeddings.feat.nodes"],
                outputs="data_release.feat.nodes_with_embeddings",
                name="ingest_nodes_with_embeddings",
            ),
            ArgoNode(
                func=ingest_nodes,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.kg_nodes",
                name="ingest_kg_nodes",
                tags=["neo4j"],
            ),
            last_node,
            # NOTE: Enable if you want embeddings
            # ArgoNode(
            #     func=lambda _, x: x,
            #     inputs=["data_release.prm.kg_nodes", "embeddings.feat.nodes"],
            #     outputs="data_release.prm.kg_embeddings",
            # )
            # need nodes that bring the nodes/edges to BigQuery
        ]
    )
