from kedro.pipeline import Pipeline, node, pipeline


from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.create_feat_nodes,
                inputs=[
                    "raw.rtx_kg2.nodes",
                    "int.embeddings",
                    "params:drug_types",
                    "params:disease_types",
                    "raw.fda_drugs",
                ],
                outputs="feat.rtx_kg2",
                name="create_feat_nodes",
            ),
            node(
                func=nodes.create_prm_pairs,
                inputs=[
                    "feat.rtx_kg2",
                    "raw.experiments.tp",
                    "raw.experiments.tn",
                ],
                outputs="prm.known_pairs",
                name="create_prm_known_pairs",
            ),
            node(
                func=nodes.make_splits,
                inputs=["prm.known_pairs", "params:splitter"],
                outputs="model_input.splits",
                name="create_splits",
            ),
            node(
                func=nodes.create_model_input_nodes,
                inputs=["feat.rtx_kg2", "model_input.splits", "params:generator"],
                outputs="model_input.enriched_splits",
                name="enrich_splits",
            ),
            node(
                func=nodes.apply_transformers,
                inputs=[
                    "model_input.enriched_splits",
                    "params:transformers",
                ],
                outputs="model_input.transformed_splits",
                name="transform_data",
            ),
            node(
                func=nodes.train_model,
                inputs={
                    "data": "model_input.transformed_splits",
                    "unpack": "params:train_args",
                },
                outputs="models.model",
                name="train_model",
            ),
        ]
    )
