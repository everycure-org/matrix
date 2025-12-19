from kedro.pipeline import Pipeline
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
    return Pipeline(
        [
            ArgoNode(
                func=ingest_nodes,
                inputs=["integration.prm.unified_nodes"],
                outputs="data_release.prm.kg_nodes",
                name="ingest_kg_nodes",
                tags=["neo4j"],
                argo_config=small_resource_requirement,
            ),
            ArgoNode(
                func=ingest_edges,
                inputs=["data_release.prm.kg_nodes", "integration.prm.unified_edges"],
                outputs="data_release.prm.kg_edges",
                name="ingest_kg_edges",
                tags=["neo4j"],
                argo_config=small_resource_requirement,
            ),
        ]
    )
