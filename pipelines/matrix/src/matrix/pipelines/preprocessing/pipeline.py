"""Preprocessing pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # Enrich curie with node synonymizer
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
            # NOTE: Filter away all nodes that we could not resolve
            # FUTURE: Either Charlotte needs to ensure things join OR
            #   We need to agree that unresolved nodes should introduce
            #   new concepts.
            node(
                func=nodes.create_prm_nodes,
                inputs=["preprocessing.int.exp.nodes@spark"],
                outputs="preprocessing.prm.exp.nodes",
                name="create_prm_exp_nodes",
                tags=["exp"],
            ),
            # Ensure edges use synonymized identifiers
            # NOTE: Charlotte introduces her own identifiers in the
            # nodes dataset, to enable edge creation.
            node(
                func=nodes.create_prm_edges,
                inputs=[
                    "preprocessing.prm.exp.nodes",
                    "preprocessing.int.exp.edges@spark",
                ],
                outputs="preprocessing.prm.exp.edges",
                name="create_prm_exp_edges",
                tags=["exp"],
            ),
            # NET: Edges dataset that connects synonymized identifiers
        ]
    )
