"""Embeddings pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # Ingest edges into Neo4j
            node(
                func=nodes.create_nodes,
                inputs=["ingestion.prm.rtx_kg2.nodes"],
                outputs="embeddings.prm.graph_nodes",
                name="create_neo4j_node_embedding_input_nodes",
                tags=["argowf.fuse-group.node_embeddings"],
            ),
            node(
                func=nodes.compute_embeddings,
                inputs={
                    "input": "embeddings.prm.graph_nodes",
                    "gdb": "params:embeddings.gdb",
                    "features": "params:embeddings.node.features",
                    "unpack": "params:embeddings.ai_config",
                },
                outputs="embeddings.prm.graph.embeddings@yaml",
                name="create_neo4j_node_embeddings",
                tags=["argowf.fuse-group.node_embeddings"],
            ),
            # Reduce dimension
            # TODO: Materialize as spark dataset
            node(
                func=nodes.reduce_dimension,
                inputs={
                    "df": "embeddings.prm.graph.embeddings@neo",
                    "unpack": "params:embeddings.dimensionality_reduction",
                },
                outputs="embeddings.feat.graph.pca_embeddings",
                name="apply_pca",
                tags=["argowf.fuse-group.node_embeddings"],
            ),
            # TODO: Needs to be new fusing group
            # Load spark dataset into local neo instance
            node(
                func=lambda x: x,
                inputs=["embeddings.feat.graph.pca_embeddings"],
                outputs="embeddings.tmp.input_nodes",
                name="ingest_neo4j_input_nodes",
                tags=["argowf.fuse-group.topological_embeddings"],
            ),
            node(
                func=nodes.ingest_edges,
                inputs=[
                    "embeddings.tmp.input_nodes",
                    "ingestion.prm.rtx_kg2.edges",
                    "params:integration.graphsage_excl_preds",
                ],
                outputs="embeddings.tmp.input_edges",
                name="ingest_neo4j_input_edges",
                tags=["argowf.fuse-group.topological_embeddings"],
            ),
            node(
                func=nodes.train_topological_embeddings,
                inputs={
                    "df": "embeddings.tmp.input_edges",
                    "gds": "params:embeddings.gds",
                    "unpack": "params:embeddings.topological",
                },
                outputs="embeddings.models.graphsage",
                name="train_topological_embeddings",
                tags=["argowf.fuse-group.topological_embeddings"],
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
                tags=["argowf.fuse-group.topological_embeddings", "argowf.mem-100g"],
            ),
            # extracts the nodes from neo4j and writes them to BigQuery
            node(
                func=lambda x: x,
                inputs=["embeddings.model_output.graphsage"],
                outputs="embeddings.feat.nodes",
                name="extract_nodes_edges_from_db",
                tags=["argowf.fuse-group.topological_embeddings"],
            ),
        ],
        tags=["argowf.fuse", "argowf.template-neo4j"],
    )
