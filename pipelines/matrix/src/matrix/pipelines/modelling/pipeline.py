"""Modelling pipeline."""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from matrix import settings

from . import nodes


def create_modeling_pipeline(**kwargs) -> Pipeline:
    """Function creates a modeling pipeline."""
    return pipeline(
        [
            node(
                func=nodes.create_model_input_nodes,
                inputs=[
                    "feat.rtx_kg2",
                    "model_input.splits",
                    "params:model_options.generator",
                ],
                outputs="model_input.enriched_splits",
                name="enrich_splits",
            ),
            node(
                func=nodes.apply_transformers,
                inputs=[
                    "model_input.enriched_splits",
                    "params:model_options.transformers",
                ],
                outputs="model_input.transformed_splits",
                name="transform_data",
            ),
            node(
                func=nodes.tune_parameters,
                inputs={
                    "data": "model_input.transformed_splits",
                    "unpack": "params:model_options.model_tuning_args",
                },
                outputs="models.model_params",  # "reporting.tuning_convergence_plot",
                name="tune_model_parameters",
            ),
            node(
                func=nodes.train_model,
                inputs={
                    "data": "model_input.transformed_splits",
                    "estimators": "models.model_params",
                    "features": "params:model_options.model_tuning_args.features",
                    "target_col_name": "params:model_options.model_tuning_args.target_col_name",
                },
                outputs="models.model",
                name="train_model",
            ),
            node(
                func=nodes.get_model_predictions,
                inputs={
                    "data": "model_input.transformed_splits",
                    "model": "models.model",
                    "features": "params:model_options.model_tuning_args.features",
                    "target_col_name": "params:model_options.model_tuning_args.target_col_name",
                },
                outputs="model_output.predictions",
                name="get_model_predictions",
            ),
            # node(
            #     func=nodes.get_model_performance,
            #     inputs={
            #         "data": "model_output.predictions",
            #         "metrics": "params:model_options.metrics",
            #         "target_col_name": "params:model_options.model_tuning_args.target_col_name",
            #     },
            #     outputs="reporting.metrics",
            #     name="get_model_performance",
            # ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create modelling pipeline."""
    create_model_input = pipeline(
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
        ]
    )

    pipes = []
    for pipeline_, namespaces in settings.DYNAMIC_PIPELINES_MAPPING.items():
        for namespace in namespaces:
            pipes.append(
                pipeline(
                    create_modeling_pipeline(),
                    inputs={
                        "feat.rtx_kg2": "modelling.feat.rtx_kg2",
                        "model_input.splits": "modelling.model_input.splits",
                    },
                    namespace=f"{pipeline_}.{namespace}",
                    tags=namespace,
                )
            )

    # consolidate = pipeline(
    #     [
    #         node(
    #             func=nodes.consolidate_reports,
    #             inputs=[
    #                 f"modelling.{namespace}.reporting.metrics"
    #                 for namespace in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    #             ],
    #             outputs="modelling.reporting.metrics",
    #             name="create_global_report",
    #             tags=[
    #                 namespace
    #                 for namespace in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    #             ],
    #         ),
    #     ]
    # )

    return sum([create_model_input, *pipes])  # consolidate
