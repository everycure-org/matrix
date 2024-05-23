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
                    "raw.ground_truth.tp",
                    "raw.ground_truth.tn",
                ],
                outputs="prm.known_pairs",
                name="create_prm_known_pairs",
            ),
            node(
                func=nodes.make_splits,
                inputs=["prm.known_pairs", "params:modelling.splitter"],
                outputs="model_input.splits",
                name="create_splits",
            ),
            node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "feat.rtx_kg2",
                    "model_input.splits",
                    "params:modelling.generator",
                ],
                outputs="model_input.enriched_splits",
                name="enrich_splits",
            ),
            node(
                func=nodes.apply_transformers,
                inputs=[
                    "model_input.enriched_splits",
                    "params:modelling.transformers",
                ],
                outputs="model_input.transformed_splits",
                name="transform_data",
            ),
            node(
                func=nodes.tune_parameters,
                inputs={
                    "data": "model_input.transformed_splits",
                    "unpack": "params:modelling.model_tuning_args",
                },
                outputs=["models.model_params", "reporting.tuning_convergence_plot"],
                name="tune_model_parameters",
            ),
            node(
                func=nodes.train_model,
                inputs={
                    "data": "model_input.transformed_splits",
                    "estimator": "models.model_params",
                    "features": "params:modelling.model_tuning_args.features",
                    "target_col_name": "params:modelling.model_tuning_args.target_col_name",
                },
                outputs="models.model",
                name="train_model",
            ),
            node(
                func=nodes.get_model_predictions,
                inputs={
                    "data": "model_input.transformed_splits",
                    "estimator": "models.model",
                    "features": "params:modelling.model_tuning_args.features",
                    "target_col_name": "params:modelling.model_tuning_args.target_col_name",
                },
                outputs="model_output.predictions",
                name="get_model_predictions",
            ),
            node(
                func=nodes.get_model_performance,
                inputs={
                    "data": "model_output.predictions",
                    "metrics": "params:modelling.metrics",
                    "target_col_name": "params:modelling.model_tuning_args.target_col_name",
                },
                outputs="reporting.metrics",
                name="get_model_performance",
            ),
        ]
    )
