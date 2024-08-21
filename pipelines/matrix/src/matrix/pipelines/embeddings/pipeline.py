"""Embeddings pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # Load data from source neo4j instance
            # Should be done after adding node embeddings
            node(
                func=lambda x: x,
                inputs=["embeddings.tmp.source_nodes"],
                outputs="embeddings.tmp.target_nodes",
                name="extract_neo4j_nodes",
            ),

            # Load spark dataset into local neo instance
            node(
                func=lambda x: x,
                inputs=["embeddings.tmp.target_nodes"],
                outputs="embeddings.tmp.input_nodes",
                name="ingest_neo4j_input_nodes",
            ),

            # Load into target neo4j instance
            # node(
            #     func=nodes.compute_embeddings,
            #     inputs={
            #         "input": "integration.model_input.nodes",
            #         "gdb": "params:embeddings.gdb",
            #         "features": "params:embeddings.node.features",
            #         "unpack": "params:embeddings.ai_config",
            #     },
            #     outputs="embeddings.prm.graph.embeddings@yaml",
            #     name="add_node_embeddings",
            #     tags=["argo.retries-3"],
            # ),
            # node(
            #     func=nodes.reduce_dimension,
            #     inputs={
            #         "df": "embeddings.prm.graph.embeddings@neo",
            #         "unpack": "params:embeddings.dimensionality_reduction",
            #     },
            #     outputs="embeddings.prm.graph.pca_embeddings",
            #     name="apply_pca",
            # ),
            node(
                func=nodes.train_topological_embeddings,
                inputs={
                    "df": "embeddings.prm.graph.pca_embeddings",
                    "edges": "integration.model_input.edges",
                    "gds": "params:embeddings.gds",
                    "unpack": "params:embeddings.topological",
                },
                outputs="embeddings.models.graphsage",
                name="train_topological_embeddings",
            ),
            node(
                func=nodes.write_topological_embeddings,
                inputs={
                    "model": "embeddings.models.graphsage",
                    "gds": "params:embeddings.gds",
                    "unpack": "params:embeddings.topological",
                },
                outputs="embeddings.model_output.graphsage",
                name="add_topological_embeddings",
            ),
            # extracts the nodes from neo4j and writes them to BigQuery
            node(
                func=nodes.extract_nodes_edges,
                inputs={
                    "nodes": "embeddings.model_output.graphsage",
                    # edges currently aren't manipulated in Neo4J thus we take input
                    "edges": "integration.model_input.edges",
                },
                outputs={
                    "enriched_nodes": "embeddings.feat.nodes",
                    "enriched_edges": "embeddings.feat.edges",
                },
                name="extract_nodes_edges_from_db",
            ),
        ],
        tags=["argo-wf.fuse", "argo-wf.enable-neo4j"],
    )
