from typing import Union

from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ARGO_GPU_NODE_MEDIUM, argo_node

from . import nodes
from .utils import partial_fold


def _create_model_shard_pipeline(model: str, shard: int, fold: Union[str, int]) -> Pipeline:
    """Create pipeline for single model, fold and shard.

    Args:
        model: model name
        shard: shard
        fold: fold to generate
    Returns:
        Pipeline with nodes for given model and fold
    """
    return pipeline(
        [
            argo_node(
                func=partial_fold(nodes.apply_transformers, fold),
                inputs={
                    "data": f"modelling.{model}.{shard}.model_input.enriched_splits",
                    "transformers": f"modelling.{model}.fold_{fold}.model_input.transformers",
                },
                outputs=f"modelling.{model}.{shard}.fold_{fold}.model_input.transformed_splits",
                name=f"transform_{model}_{shard}_data_fold_{fold}",
            ),
            argo_node(
                func=nodes.tune_parameters,
                inputs={
                    "data": f"modelling.{model}.{shard}.fold_{fold}.model_input.transformed_splits",
                    "unpack": f"params:modelling.{model}.model_options.model_tuning_args",
                },
                outputs=[
                    f"modelling.{model}.{shard}.fold_{fold}.models.model_params",
                    f"modelling.{model}.{shard}.fold_{fold}.reporting.tuning_convergence_plot",
                ],
                name=f"tune_model_{model}_{shard}_parameters_fold_{fold}",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
            argo_node(
                func=nodes.train_model,
                inputs=[
                    f"modelling.{model}.{shard}.fold_{fold}.model_input.transformed_splits",
                    f"modelling.{model}.{shard}.fold_{fold}.models.model_params",
                    f"params:modelling.{model}.model_options.model_tuning_args.features",
                    f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                ],
                outputs=f"modelling.{model}.{shard}.fold_{fold}.models.model",
                name=f"train_{model}_{shard}_model_fold_{fold}",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.shard-{shard}.fold-{fold}"],
    )


def _create_fold_pipeline(model: str, num_shards: int, fold: Union[str, int]) -> Pipeline:
    """Create pipeline for single model and fold.

    Args:
        model: model name
        num_shards: number of shards to generate
        fold: fold to generate
    Returns:
        Pipeline with nodes for given model and fold
    """
    return sum(
        [
            pipeline(
                [
                    argo_node(
                        func=partial_fold(nodes.fit_transformers, fold),
                        inputs={
                            "data": "modelling.model_input.splits",
                            "transformers": f"params:modelling.{model}.model_options.transformers",
                        },
                        outputs=f"modelling.{model}.fold_{fold}.model_input.transformers",
                        name=f"fit_{model}_transformers_fold_{fold}",
                        tags=model,
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    )
                ]
            ),
            *[
                pipeline(
                    _create_model_shard_pipeline(model=model, shard=shard, fold=fold),
                    tags=model,
                )
                for shard in range(num_shards)
            ],
            pipeline(
                [
                    argo_node(
                        func=nodes.create_model,
                        inputs=[f"params:modelling.{model}.model_options.ensemble.agg_func"]
                        + [f"modelling.{model}.{shard}.fold_{fold}.models.model" for shard in range(num_shards)],
                        outputs=f"modelling.{model}.fold_{fold}.models.model",
                        name=f"create_{model}_model_fold_{fold}",
                        tags=model,
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    ),
                    argo_node(
                        func=partial_fold(nodes.apply_transformers, fold),
                        inputs={
                            "data": "modelling.model_input.splits",
                            "transformers": f"modelling.{model}.fold_{fold}.model_input.transformers",
                        },
                        outputs=f"modelling.{model}.fold_{fold}.model_input.transformed_splits",
                        name=f"transform_{model}_data_fold_{fold}",
                    ),
                    argo_node(
                        func=nodes.get_model_predictions,
                        inputs={
                            "data": f"modelling.{model}.fold_{fold}.model_input.transformed_splits",
                            "model": f"modelling.{model}.fold_{fold}.models.model",
                            "features": f"params:modelling.{model}.model_options.model_tuning_args.features",
                            "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.{model}.fold_{fold}.model_output.predictions",
                        name=f"get_{model}_model_predictions_fold_{fold}",
                        argo_config=ARGO_GPU_NODE_MEDIUM,
                    ),
                ],
                tags=["argowf.fuse", f"argowf.fuse-group.{model}.fold-{fold}"],
            ),
        ]
    )


def create_model_pipeline(model: str, num_shards: int, n_cross_val_folds: int) -> Pipeline:
    """Create pipeline for a single model.

    Args:
        model: model name
        num_shards: number of shard to generate
        n_cross_val_folds: number of folds for cross-validation (i.e. number of test/train splits, not including fold with full training data)
    Returns:
        Pipeline with model nodes
    """
    pipelines = []

    # Generate pipeline to enrich splits
    pipelines.append(
        pipeline(
            [
                argo_node(
                    func=nodes.create_model_input_nodes,
                    inputs=[
                        "modelling.model_input.drugs_diseases_nodes@pandas",
                        "modelling.model_input.splits",
                        f"params:modelling.{model}.model_options.generator",
                    ],
                    outputs=f"modelling.{model}.{shard}.model_input.enriched_splits",
                    name=f"enrich_{model}_{shard}_splits",
                )
                for shard in range(num_shards)
            ]
        )
    )

    # Generate pipeline to predict folds (NOTE: final fold is full training data)
    for fold in range(n_cross_val_folds + 1):
        pipelines.append(
            pipeline(
                _create_fold_pipeline(model=model, num_shards=num_shards, fold=fold),
                tags=[model, "not-shared"],
            )
        )

    # Gather all test set predictions from the different folds for the
    # model, and combine all the predictions.
    pipelines.append(
        pipeline(
            [
                argo_node(
                    func=nodes.combine_data,
                    inputs=[
                        f"modelling.{model}.fold_{fold}.model_output.predictions" for fold in range(n_cross_val_folds)
                    ],
                    outputs=f"modelling.{model}.model_output.combined_predictions",
                    name=f"combine_{model}_folds",
                    tags=[f"{model}"],
                )
            ]
        )
    )

    # Now check the performance on the combined folds
    pipelines.append(
        pipeline(
            [
                argo_node(
                    func=nodes.check_model_performance,
                    inputs={
                        "data": f"modelling.{model}.model_output.combined_predictions",
                        "metrics": f"params:modelling.{model}.model_options.metrics",
                        "target_col_name": f"params:modelling.{model}.model_options.model_tuning_args.target_col_name",
                    },
                    outputs=f"modelling.{model}.reporting.metrics",
                    name=f"check_{model}_model_performance",
                    tags=[f"{model}"],
                )
            ]
        )
    )

    return sum(pipelines)


def create_shared_pipeline(model_name: str) -> Pipeline:
    """Function to create pipeline of shared nodes.

    NOTE: The model list is added to tag the nodes for single pipeline execution.

    Args:
        models_lst: list of models to generate
    Returns:
        Pipeline with shared nodes across models
    """
    return pipeline(
        [
            # Construct ground_truth
            argo_node(
                func=nodes.filter_valid_pairs,
                inputs=[
                    "integration.prm.filtered_nodes",
                    "modelling.raw.ground_truth.positives@spark",
                    "modelling.raw.ground_truth.negatives@spark",
                ],
                outputs={"pairs": "modelling.raw.known_pairs@spark", "metrics": "modelling.reporting.gt_present"},
                name="filter_valid_pairs",
            ),
            argo_node(
                func=nodes.attach_embeddings,
                inputs=[
                    "modelling.raw.known_pairs@spark",
                    "embeddings.feat.nodes",
                ],
                outputs="modelling.int.known_pairs@spark",
                name="create_int_known_pairs",
            ),
            argo_node(
                func=nodes.prefilter_nodes,
                inputs=[
                    "integration.prm.filtered_nodes",
                    "embeddings.feat.nodes",
                    "modelling.raw.known_pairs@spark",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="modelling.model_input.drugs_diseases_nodes@spark",
                name="prefilter_nodes",
            ),
            argo_node(
                func=nodes.make_folds,
                inputs=[
                    "modelling.int.known_pairs@pandas",
                    "params:modelling.splitter",
                ],
                outputs="modelling.model_input.splits",
                name="create_splits",
            ),
        ],
        tags=model_name,
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create modelling pipeline.

    FUTURE: Try cleanup step where folds are passed in using partials, to ensure
    we can keep a single dataframe of fold information.

    Pipeline is created dynamically, based on the following dimensions:
        - Models, i.e., type of model, e.g. random forest
        - Folds, i.e., number of folds to train/evaluation
        - Shards, i.e., defined for ensemble models, non-ensemble models have shards = 1
    """
    # Unpack model
    model = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")
    model_name = model["model_name"]
    model_config = model["model_config"]

    # Unpack Folds
    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_cross_val_folds")

    # Add shared nodes
    pipelines = []
    pipelines.append(create_shared_pipeline(model_name))

    # Generate pipeline for the model
    pipelines.append(create_model_pipeline(model_name, model_config["num_shards"], n_cross_val_folds))

    return sum(pipelines)
