"""Preprocessing pipeline."""
from functools import partial
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


# NOTE: This pipeline in highly preliminary and used for ingestion of the
# medical data provided in Google Sheets __ONLY__.
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
                name="resolve_ec_medical_team_nodes",
            ),
            # NOTE: Running this to get the identifiers in the KG
            # Normalize nodes
            node(
                func=partial(
                    nodes.enrich_df,
                    func=nodes.normalize,
                    input_cols=["corrected_curie", "curie"],
                    target_col="normalized_curie",
                    coalesce_col="new_id",
                ),
                inputs=[
                    "preprocessing.int.resolved_nodes",
                    "params:preprocessing.synonymizer_endpoint",
                ],
                outputs="preprocessing.int.normalized_nodes",
                name="normalize_ec_medical_team_nodes",
            ),
            # NOTE: Filter away all nodes that we could not resolve
            # FUTURE: Either Charlotte needs to ensure things join OR
            #   We need to agree that unresolved nodes should introduce
            #   new concepts.
            node(
                func=nodes.create_int_nodes,
                inputs=["preprocessing.int.normalized_nodes"],
                outputs="preprocessing.int.nodes",
                name="create_int_ec_medical_team_nodes",
            ),
            # Ensure edges use synonymized identifiers
            # NOTE: Charlotte introduces her own identifiers in the
            # nodes dataset, to enable edge creation.
            node(
                func=nodes.create_int_edges,
                inputs=[
                    "preprocessing.int.nodes",
                    "preprocessing.raw.edges",
                ],
                outputs="preprocessing.int.edges",
                name="create_int_ec_medical_team_edges",
            ),
            node(
                func=nodes.create_prm_edges,
                inputs=[
                    "preprocessing.int.edges",
                ],
                outputs="ingestion.raw.ec_medical_team.edges@pandas",
                name="create_prm_ec_medical_team_edges",
            ),
            node(
                func=nodes.create_prm_nodes,
                inputs=[
                    "preprocessing.int.nodes",
                ],
                outputs="ingestion.raw.ec_medical_team.nodes@pandas",
                name="create_prm_ec_medical_team_nodes",
            ),
        ]
    )
