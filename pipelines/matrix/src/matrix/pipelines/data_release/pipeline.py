"""Pipeline to release data."""

from kedro.pipeline import Pipeline, node, pipeline
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
            # write to BigQuery
            node(
                func=lambda x: x,
                inputs=["integration.prm.filtered_edges"],
                outputs="data_release.prm.bigquery_edges",
                name="write_edges_to_bigquery",
                tags=["bigquery"],
            ),
            node(
                func=lambda x: x,
                inputs=["integration.prm.filtered_nodes"],
                outputs="data_release.prm.bigquery_nodes",
                name="write_nodes_to_bigquery",
                tags=["bigquery"],
            ),
        ]
    )
