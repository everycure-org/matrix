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
                outputs=[
                    "modelling.model_input.transformed_splits",
                    "modelling.model_input.transformers" 
                ],
                name="transform_data",
            ),
            node(
                func=nodes.tune_parameters,
                inputs={
                    "data": "modelling.model_input.transformed_splits",
                    "unpack": "params:modelling.model_tuning_args",
                },
                outputs=[
                    "modelling.models.model_params",
                    "modelling.reporting.tuning_convergence_plot",
                ],
                name="tune_model_parameters",
            ),
            node(
                func=nodes.train_model,
                inputs={
                    "data": "modelling.model_input.transformed_splits",
                    "estimator": "modelling.models.model_params",
                    "features": "params:modelling.model_tuning_args.features",
                    "target_col_name": "params:modelling.model_tuning_args.target_col_name",
                },
                outputs="modelling.models.model",
                name="train_model",
            ),
            node(
                func=nodes.generate_drp_model,
                inputs={
                    "estimator": "modelling.models.model",
                    "graph": "modelling.feat.rtx_kg2",
                    "transformers": "modelling.model_input.transformers",
                    "features": "params:modelling.model_tuning_args.features",
                },
                outputs="modelling.models.drp_model",
                name="get_model_predictions",
            ),
            node(
                func=nodes.get_classification_metrics,
                inputs={
                    "drp_model": "modelling.models.drp_model", 
                    "data": "modelling.model_input.transformed_splits",
                    "metrics": "params:modelling.metrics",
                    "target_col_name": "params:modelling.model_tuning_args.target_col_name",
                },
                outputs="modelling.reporting.classification_metrics",
                name="get_classification_metrics",
            ),
            node(
                func=nodes.perform_disease_centric_evaluation,
                inputs={
                    "drp_model": "modelling.models.drp_model", 
                    "known_data": "modelling.model_input.transformed_splits",
                },
                outputs="modelling.reporting.ranking_metrics",
                name="get_model_performance",
            ),
        ]
    )
