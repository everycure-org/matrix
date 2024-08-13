"""Preprocessing pipeline."""
from functools import partial
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            # NOTE: Running this to get an initial proposal of curies
            # Enrich curie with node synonymizer
            node(
                func=partial(
                    nodes.enrich_df,
                    func=nodes.resolve,
                    input_cols=["name"],
                    target_col="curie",
                ),
                inputs=[
                    "preprocessing.raw.nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.resolved_nodes",
                name="resolve_nodes",
            ),
            # NOTE: Running this to get the identifiers in the KG
            # Normalize nodes
            node(
                func=partial(
                    nodes.enrich_df,
                    func=nodes.normalize,
                    input_cols=["corrected_curie", "curie"],
                    target_col="normalized_curie",
                ),
                inputs=[
                    "preprocessing.int.resolved_nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.normalized_nodes",
                name="normalize_nodes",
            ),
            # NOTE: Filter away all nodes that we could not resolve
            # FUTURE: Either Charlotte needs to ensure things join OR
            #   We need to agree that unresolved nodes should introduce
            #   new concepts.
            node(
                func=nodes.create_prm_nodes,
                inputs=["preprocessing.int.nodes@spark"],
                outputs="preprocessing.prm.nodes",
                name="create_prm_nodes",
            ),
            # node(
            #     func=lambda x: x,
            #     inputs=["preprocessing.raw.edges"],
            #     outputs="preprocessing.int.edges@pandas",
            #     name="create_int_edges",
            # ),
            # # Ensure edges use synonymized identifiers
            # # NOTE: Charlotte introduces her own identifiers in the
            # # nodes dataset, to enable edge creation.
            # node(
            #     func=nodes.create_prm_edges,
            #     inputs=[
            #         "preprocessing.prm.nodes",
            #         "preprocessing.int.edges@spark",
            #     ],
            #     outputs="preprocessing.prm.edges",
            #     name="create_prm_exp_edges",
            # ),
        ]
    )
