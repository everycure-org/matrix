"""Embeddings pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            node(
                func=nodes.compute_embeddings,
                inputs={
                    "input": "integration.model_input.nodes",
                    "gdb": "params:embeddings.gdb",
                    "features": "params:embeddings.node.features",
                    "unpack": "params:embeddings.ai_config",
                },
                outputs="embeddings.prm.graph.embeddings@yaml",
                name="add_node_embeddings",
                tags=["argo.retries-3"],
            ),
            node(
                func=nodes.reduce_dimension,
                inputs=[
                    "embeddings.prm.graph.embeddings@neo",
                    "params:embeddings.dimensionality_reduction",
                ],
                outputs="embeddings.prm.graph.pca_embeddings",
                name="apply_pca",
            ),
            node(
                func=nodes.add_topological_embeddings,
                inputs={
                    "df": "embeddings.prm.graph.pca_embeddings",
                    "edges": "integration.model_input.edges",
                    "gds": "params:embeddings.gds",
                    "unpack": "params:embeddings.topological",
                },
                outputs="embeddings.model_output.graphsage",
                name="add_topological_embeddings",
            ),
        ]
    )
