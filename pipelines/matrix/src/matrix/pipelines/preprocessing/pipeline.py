"""Preprocessing pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=nodes.resolve_nodes,
                inputs=[
                    "preprocessing.raw.exp.nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.exp.nodes@pandas",
                name="resolve_exp_nodes",
                tags=["exp"],
            ),
            node(
                func=lambda x: x,
                inputs=["preprocessing.raw.exp.edges"],
                outputs="preprocessing.int.exp.edges@pandas",
                name="create_int_edges",
                tags=["exp"],
            ),
            node(
                func=nodes.create_prm_nodes,
                inputs=["preprocessing.int.exp.nodes@spark"],
                outputs="preprocessing.prm.exp.nodes",
                name="create_prm_nodes",
                tags=["exp"],
            ),
            node(
                func=nodes.create_prm_edges,
                inputs=[
                    "preprocessing.prm.exp.nodes",
                    "preprocessing.int.exp.edges@spark",
                ],
                outputs="preprocessing.prm.exp.edges",
                name="create_prm_edges",
                tags=["exp"],
            ),
        ]
    )
