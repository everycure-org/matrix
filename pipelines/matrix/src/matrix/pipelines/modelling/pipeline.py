"""Modelling pipeline."""
from kedro.pipeline import Pipeline, node, pipeline


from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create modelling pipeline."""
    return pipeline(
        [
            node(
                func=nodes.create_feat_nodes,
                inputs=[
                    "modelling.raw.rtx_kg2.nodes",
                    "embeddings.int.graphsage",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                    "modelling.raw.fda_drugs",
                ],
                outputs="modelling.feat.rtx_kg2",
                name="create_feat_nodes",
            ),
            node(
                func=nodes.create_prm_pairs,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "modelling.raw.ground_truth.tp",
                    "modelling.raw.ground_truth.tn",
                ],
                outputs="modelling.prm.known_pairs",
                name="create_prm_known_pairs",
            ),
            node(
                func=nodes.make_splits,
                inputs=["modelling.prm.known_pairs", "params:modelling.splitter"],
                outputs="modelling.model_input.splits",
                name="create_splits",
            ),
            node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "modelling.feat.rtx_kg2",
                    "modelling.model_input.splits",
                    "params:modelling.generator",
                ],
                outputs="modelling.model_input.enriched_splits",
                name="enrich_splits",
            ),
            node(
                func=nodes.apply_transformers,
                inputs=[
                    "modelling.model_input.enriched_splits",
                    "params:modelling.transformers",
                ],
                outputs="modelling.model_input.transformed_splits",
                name="transform_data",
            ),
            node(
                func=nodes.train_model,
                inputs={
                    "data": "modelling.model_input.transformed_splits",
                    "unpack": "params:modelling.train_args",
                },
                outputs="modelling.models.model",
                name="train_model",
            ),
        ]
    )
