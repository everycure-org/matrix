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
                    input_col="name",
                    target_col="curie",
                ),
                inputs=[
                    "preprocessing.raw.nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.nodes",
                name="resolve_nodes",
                tags=["resolve"],
            ),
            # NOTE: Running this to get the identifiers in the KG
            # Normalize nodes
            node(
                func=partial(
                    nodes.enrich_df,
                    func=nodes.normalize,
                    input_col="curie",
                    target_col="id",
                ),
                inputs=[
                    "preprocessing.int.nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.prm.nodes",
                name="normalize_nodes",
                tags=["normalize"],
            ),
            # # Transcode to pandas
            # node(
            #     func=lambda x: x,
            #     inputs=["preprocessing.int.exp.nodes_excel"],
            #     outputs="preprocessing.int.exp.nodes@pandas",
            #     name="transcode_pandas",
            #     tags=["exp"],
            # ),
            # node(
            #     func=lambda x: x,
            #     inputs=["preprocessing.raw.exp.edges"],
            #     outputs="preprocessing.int.exp.edges@pandas",
            #     name="create_int_edges",
            #     tags=["exp"],
            # ),
            # # NOTE: Filter away all nodes that we could not resolve
            # # FUTURE: Either Charlotte needs to ensure things join OR
            # #   We need to agree that unresolved nodes should introduce
            # #   new concepts.
            # node(
            #     func=nodes.create_prm_nodes,
            #     inputs=["preprocessing.int.exp.nodes@spark"],
            #     outputs="preprocessing.prm.exp.nodes",
            #     name="create_prm_exp_nodes",
            #     tags=["exp"],
            # ),
            # # Ensure edges use synonymized identifiers
            # # NOTE: Charlotte introduces her own identifiers in the
            # # nodes dataset, to enable edge creation.
            # node(
            #     func=nodes.create_prm_edges,
            #     inputs=[
            #         "preprocessing.prm.exp.nodes",
            #         "preprocessing.int.exp.edges@spark",
            #     ],
            #     outputs="preprocessing.prm.exp.edges",
            #     name="create_prm_exp_edges",
            #     tags=["exp"],
            # ),
            # # NET: Edges dataset that connects synonymized identifiers
        ]
    )
