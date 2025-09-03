from kedro.pipeline import Pipeline, pipeline
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from matrix.pipelines.embeddings.nodes import ingest_edges, ingest_nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingest to Neo4J pipeline."""
    small_resource_requirement = ArgoResourceConfig(
        cpu_limit=2,
        cpu_request=2,
        memory_limit=24,
        memory_request=24,
    )
    return pipeline(
        [
            ArgoNode(
                func=ingest_nodes,
                inputs=["filtering.prm.filtered_nodes"],
                outputs="data_release.prm.kg_nodes",
                name="ingest_kg_nodes",
                tags=["neo4j"],
                argo_config=small_resource_requirement,
            ),
            ArgoNode(
                func=ingest_edges,
                inputs=["data_release.prm.kg_nodes", "filtering.prm.filtered_edges"],
                outputs="data_release.prm.kg_edges",
                name="ingest_kg_edges",
                tags=["neo4j"],
                argo_config=small_resource_requirement,
            ),
        ]
    )
