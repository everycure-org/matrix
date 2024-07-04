"""Embeddings pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            node(
                func=nodes.concat_features,
                inputs=[
                    "integration.model_input.nodes",
                    "params:embeddings.node.features",
                    "params:embeddings.ai_config",
                ],
                outputs="embeddings.prm.graph.embeddings",
                name="add_node_embeddings",
                tags=["argo:retries=3"],
            ),
            node(
                func=nodes.reduce_dimension,
                inputs=[
                    "embeddings.prm.graph.embeddings",
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
