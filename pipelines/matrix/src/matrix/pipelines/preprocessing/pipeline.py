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
                    "preprocessing.raw.exp.nodes@pandas",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.exp.nodes@pandas",
                name="resolve_exp_nodes",
                tags=["exp"],
            ),
        ]
    )
