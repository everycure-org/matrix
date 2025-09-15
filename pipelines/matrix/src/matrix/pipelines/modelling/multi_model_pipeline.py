from typing import Union

from kedro.pipeline import Pipeline, pipeline

from matrix.kedro4argo_node import ARGO_CPU_ONLY_NODE_MEDIUM, ARGO_GPU_NODE_MEDIUM, ArgoNode

from . import nodes
from .utils import partial_fold


def create_multi_model_pipeline(models: list[dict], n_cross_val_folds: int) -> Pipeline:
    """Create pipeline for multiple models.

    Args:
        model_name: model name  to pull the right model parameters
        num_shards: number of shard to generate
        n_cross_val_folds: number of folds for cross-validation (i.e. number of test/train splits, not including fold with full training data)
    Returns:
        Pipeline with model nodes
    """
    pipelines = []

    for model in models:
        model_name = model["model_name"]
        model_config = model["model_config"]
        num_shards = model_config.get("num_shards", 1)
        # Generate pipeline to enrich splits
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.create_model_input_nodes,
                        inputs=[
                            "modelling.model_input.drugs_diseases_nodes@pandas",
                            "modelling.model_input.splits@pandas",
                            f"params:modelling.{model_name}.model_options.generator",
                            f"params:modelling.{model_name}.splitter",
                        ],
                        outputs=f"modelling.{shard}.{model_name}.model_input.enriched_splits",
                        name=f"enrich_{model_name}_{shard}_splits",
                    )
                    for shard in range(num_shards)
                ]
            )
        )

        # Generate pipeline to predict folds (NOTE: final fold is full training data)
        for fold in range(n_cross_val_folds + 1):
            pipelines.append(pipeline(_create_multi_model_fold_pipeline(model_name, num_shards, fold)))

        # Gather all test set predictions from the different folds for the
        # model, and combine all the predictions.
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.combine_data,
                        inputs=[
                            f"modelling.fold_{fold}.{model_name}.model_output.predictions"
                            for fold in range(n_cross_val_folds)
                        ],
                        outputs=f"modelling.{model_name}.model_output.combined_predictions",
                        name=f"combine_folds_{model_name}",
                    )
                ]
            )
        )

        # # Now check the performance on the combined folds
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.check_model_performance,
                        inputs={
                            "data": f"modelling.{model_name}.model_output.combined_predictions",
                            "metrics": f"params:modelling.{model_name}.model_options.metrics",
                            "target_col_name": f"params:modelling.{model_name}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model_name}.reporting.metrics",
                        name=f"check_{model_name}_model_performance",
                    )
                ]
            )
        )

    return sum(pipelines)


def _create_multi_model_shard_pipeline(model_name: str, shard: int, fold: Union[str, int]) -> Pipeline:
    """Create pipeline for single model, fold and shard.

    Args:
        model_name: model name
        shard: shard
        fold: fold to generate
    Returns:
        Pipeline with nodes for given model and fold
    """
    return pipeline(
        [
            ArgoNode(
                func=partial_fold(nodes.apply_transformers, fold),
                inputs={
                    "data": f"modelling.{shard}.{model_name}.model_input.enriched_splits",
                    "transformers": f"modelling.fold_{fold}.{model_name}.model_input.transformers",
                },
                outputs=f"modelling.{shard}.fold_{fold}.{model_name}.model_input.transformed_splits",
                name=f"transform_{model_name}_{shard}_data_fold_{fold}",
            ),
            ArgoNode(
                func=nodes.tune_parameters,
                inputs={
                    "data": f"modelling.{shard}.fold_{fold}.{model_name}.model_input.transformed_splits",
                    "unpack": f"params:modelling.{model_name}.model_options.model_tuning_args",
                },
                outputs=[
                    f"modelling.{shard}.fold_{fold}.{model_name}.models.model_params",
                    f"modelling.{shard}.fold_{fold}.{model_name}.reporting.tuning_convergence_plot",
                ],
                name=f"tune_{model_name}_model_{shard}_parameters_fold_{fold}",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
            ArgoNode(
                func=nodes.train_model,
                inputs=[
                    f"modelling.{shard}.fold_{fold}.{model_name}.model_input.transformed_splits",
                    f"modelling.{shard}.fold_{fold}.{model_name}.models.model_params",
                    f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                    f"params:modelling.{model_name}.model_options.model_tuning_args.target_col_name",
                ],
                outputs=f"modelling.{shard}.fold_{fold}.{model_name}.models.model",
                name=f"train_{shard}_{model_name}_model_fold_{fold}",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model_name}.shard-{shard}.fold-{fold}"],
    )


def _create_multi_model_fold_pipeline(model_name: str, num_shards: int, fold: Union[str, int]) -> Pipeline:
    """Create pipeline for single model and fold.

    Args:
        model_name: model name
        num_shards: number of shards to generate
        fold: fold to generate
    Returns:
        Pipeline with nodes for given model and fold
    """
    return sum(
        [
            pipeline(
                [
                    ArgoNode(
                        func=partial_fold(nodes.fit_transformers, fold),
                        inputs={
                            "data": "modelling.model_input.splits@pandas",
                            "transformers": f"params:modelling.{model_name}.model_options.transformers",
                        },
                        outputs=f"modelling.fold_{fold}.{model_name}.model_input.transformers",
                        name=f"fit_{model_name}_transformers_fold_{fold}",
                        argo_config=ARGO_CPU_ONLY_NODE_MEDIUM,
                    )
                ]
            ),
            *[
                pipeline(
                    _create_multi_model_shard_pipeline(model_name, shard, fold),
                )
                for shard in range(num_shards)
            ],
            pipeline(
                [
                    ArgoNode(
                        func=nodes.create_model,
                        inputs=[f"params:modelling.{model_name}.model_options.ensemble.agg_func"]
                        + [f"modelling.{shard}.fold_{fold}.{model_name}.models.model" for shard in range(num_shards)],
                        outputs=f"modelling.fold_{fold}.{model_name}.models.model",
                        name=f"create_{model_name}_model_fold_{fold}",
                        argo_config=ARGO_CPU_ONLY_NODE_MEDIUM,
                    ),
                    ArgoNode(
                        func=partial_fold(nodes.apply_transformers, fold),
                        inputs={
                            "data": "modelling.model_input.splits@pandas",
                            "transformers": f"modelling.fold_{fold}.{model_name}.model_input.transformers",
                        },
                        outputs=f"modelling.fold_{fold}.{model_name}.model_input.transformed_splits",
                        name=f"transform_{model_name}_data_fold_{fold}",
                    ),
                    ArgoNode(
                        func=nodes.get_model_predictions,
                        inputs={
                            "data": f"modelling.fold_{fold}.{model_name}.model_input.transformed_splits",
                            "model": f"modelling.fold_{fold}.models.model",
                            "features": f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                            "target_col_name": f"params:modelling.{model_name}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.fold_{fold}.{model_name}.model_output.predictions",
                        name=f"get_{model_name}_model_predictions_fold_{fold}",
                        argo_config=ARGO_CPU_ONLY_NODE_MEDIUM,
                    ),
                ],
                tags=["argowf.fuse", f"argowf.fuse-group.{model_name}.fold-{fold}"],
            ),
        ]
    )
