from kedro.pipeline import Pipeline, node, pipeline
from matrix.pipelines.embeddings.nodes import ingest_edges, ingest_nodes
from matrix.tags import NodeTags


def create_pipeline(**kwargs) -> Pipeline:
    """Create release pipeline."""
    return pipeline(
        [
            node(
                func=ingest_nodes,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.kg_nodes",
                name="ingest_kg_nodes",
                tags=[NodeTags.NEO4J.value],
            ),
            node(
                func=ingest_edges,
                inputs=["data_release.prm.kg_nodes", "integration.prm.filtered_edges"],
                outputs="data_release.prm.kg_edges",
                name="ingest_kg_edges",
                tags=[NodeTags.NEO4J.value],
            ),
            # write to BigQuery
            node(
                func=lambda x: x,
                inputs=["integration.prm.filtered_edges"],
                outputs="data_release.prm.bigquery_edges",
                name="write_edges_to_bigquery",
                tags=[NodeTags.BIGQUERY.value],
            ),
            node(
                func=lambda x: x,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.bigquery_nodes",
                name="write_nodes_to_bigquery",
                tags=[NodeTags.BIGQUERY.value],
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
