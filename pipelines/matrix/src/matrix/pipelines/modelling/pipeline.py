from typing import Union

from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.kedro4argo_node import ARGO_CPU_ONLY_NODE_MEDIUM, ARGO_GPU_NODE_MEDIUM, ArgoNode

from . import nodes
from .utils import partial_fold


def _create_model_shard_pipeline(model_name: str, shard: int, fold: Union[str, int]) -> Pipeline:
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
                    "data": f"modelling.{shard}.model_input.enriched_splits",
                    "transformers": f"modelling.fold_{fold}.model_input.transformers",
                },
                outputs=f"modelling.{shard}.fold_{fold}.model_input.transformed_splits",
                name=f"transform_{shard}_data_fold_{fold}",
            ),
            ArgoNode(
                func=nodes.tune_parameters,
                inputs={
                    "data": f"modelling.{shard}.fold_{fold}.model_input.transformed_splits",
                    "unpack": f"params:modelling.{model_name}.model_options.model_tuning_args",
                },
                outputs=[
                    f"modelling.{shard}.fold_{fold}.models.model_params",
                    f"modelling.{shard}.fold_{fold}.reporting.tuning_convergence_plot",
                ],
                name=f"tune_model_{shard}_parameters_fold_{fold}",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
            ArgoNode(
                func=nodes.train_model,
                inputs=[
                    f"modelling.{shard}.fold_{fold}.model_input.transformed_splits",
                    f"modelling.{shard}.fold_{fold}.models.model_params",
                    f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                    f"params:modelling.{model_name}.model_options.model_tuning_args.target_col_name",
                ],
                outputs=f"modelling.{shard}.fold_{fold}.models.model",
                name=f"train_{shard}_model_fold_{fold}",
                argo_config=ARGO_GPU_NODE_MEDIUM,
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.shard-{shard}.fold-{fold}"],
    )


def _create_fold_pipeline(model_name: str, num_shards: int, fold: Union[str, int]) -> Pipeline:
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
                        outputs=f"modelling.fold_{fold}.model_input.transformers",
                        name=f"fit_transformers_fold_{fold}",
                        argo_config=ARGO_CPU_ONLY_NODE_MEDIUM,
                    )
                ]
            ),
            *[
                pipeline(
                    _create_model_shard_pipeline(model_name, shard, fold),
                )
                for shard in range(num_shards)
            ],
            pipeline(
                [
                    ArgoNode(
                        func=nodes.create_model,
                        inputs=[f"params:modelling.{model_name}.model_options.ensemble.agg_func"]
                        + [f"modelling.{shard}.fold_{fold}.models.model" for shard in range(num_shards)],
                        outputs=f"modelling.fold_{fold}.models.model",
                        name=f"create_model_fold_{fold}",
                        argo_config=ARGO_CPU_ONLY_NODE_MEDIUM,
                    ),
                    ArgoNode(
                        func=partial_fold(nodes.apply_transformers, fold),
                        inputs={
                            "data": "modelling.model_input.splits@pandas",
                            "transformers": f"modelling.fold_{fold}.model_input.transformers",
                        },
                        outputs=f"modelling.fold_{fold}.model_input.transformed_splits",
                        name=f"transform_data_fold_{fold}",
                    ),
                    ArgoNode(
                        func=nodes.get_model_predictions,
                        inputs={
                            "data": f"modelling.fold_{fold}.model_input.transformed_splits",
                            "model": f"modelling.fold_{fold}.models.model",
                            "features": f"params:modelling.{model_name}.model_options.model_tuning_args.features",
                            "target_col_name": f"params:modelling.{model_name}.model_options.model_tuning_args.target_col_name",
                        },
                        outputs=f"modelling.fold_{fold}.model_output.predictions",
                        name=f"get_model_predictions_fold_{fold}",
                        argo_config=ARGO_CPU_ONLY_NODE_MEDIUM,
                    ),
                ],
                tags=["argowf.fuse", f"argowf.fuse-group.fold-{fold}"],
            ),
        ]
    )


def create_model_pipeline(model_name: str, num_shards: int, n_cross_val_folds: int) -> Pipeline:
    """Create pipeline for a single model.

    Args:
        model_name: model name  to pull the right model parameters
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
                ArgoNode(
                    func=nodes.create_model_input_nodes,
                    inputs=[
                        "modelling.model_input.drugs_diseases_nodes@pandas",
                        "modelling.model_input.splits@pandas",
                        f"params:modelling.{model_name}.model_options.generator",
                        "params:modelling.splitter",
                    ],
                    outputs=f"modelling.{shard}.model_input.enriched_splits",
                    name=f"enrich_{shard}_splits",
                )
                for shard in range(num_shards)
            ]
        )
    )

    # Generate pipeline to predict folds (NOTE: final fold is full training data)
    for fold in range(n_cross_val_folds + 1):
        pipelines.append(pipeline(_create_fold_pipeline(model_name, num_shards, fold)))

    # Gather all test set predictions from the different folds for the
    # model, and combine all the predictions.
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.combine_data,
                    inputs=[f"modelling.fold_{fold}.model_output.predictions" for fold in range(n_cross_val_folds)],
                    outputs="modelling.model_output.combined_predictions",
                    name=f"combine_folds",
                )
            ]
        )
    )

    # Now check the performance on the combined folds
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.check_model_performance,
                    inputs={
                        "data": "modelling.model_output.combined_predictions",
                        "metrics": f"params:modelling.{model_name}.model_options.metrics",
                        "target_col_name": f"params:modelling.{model_name}.model_options.model_tuning_args.target_col_name",
                    },
                    outputs="modelling.reporting.metrics",
                    name="check_model_performance",
                )
            ]
        )
    )

    return sum(pipelines)


def create_shared_pipeline() -> Pipeline:
    """Function to create pipeline of shared nodes.

    NOTE: The model list is added to tag the nodes for single pipeline execution.

    Returns:
        Pipeline with shared nodes across models
    """
    return pipeline(
        [
            # Construct ground_truth
            ArgoNode(
                func=nodes.filter_valid_pairs,
                inputs=[
                    "filtering.prm.filtered_nodes",
                    "integration.prm.unified_ground_truth_edges",
                    "params:modelling.training_data_sources",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs={"pairs": "modelling.raw.known_pairs@spark", "metrics": "modelling.reporting.gt_present"},
                name="filter_valid_pairs",
            ),
            ArgoNode(
                func=nodes.attach_embeddings,
                inputs=[
                    "modelling.raw.known_pairs@spark",
                    "embeddings.feat.nodes",
                ],
                outputs="modelling.int.known_pairs@spark",
                name="create_int_known_pairs",
            ),
            ArgoNode(
                func=nodes.prefilter_nodes,
                inputs=[
                    "embeddings.feat.nodes",
                    "modelling.raw.known_pairs@spark",
                    "params:modelling.drug_types",
                    "params:modelling.disease_types",
                ],
                outputs="modelling.model_input.drugs_diseases_nodes@spark",
                name="prefilter_nodes",
            ),
            ArgoNode(
                func=nodes.make_folds,
                inputs=[
                    "modelling.int.known_pairs@pandas",
                    "params:modelling.splitter",
                    "integration.int.disease_list.nodes.norm@pandas",
                ],
                outputs="modelling.model_input.splits@pandas",
                name="create_splits",
            ),
        ]
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
    model = settings.DYNAMIC_PIPELINES_MAPPING().get("modelling")
    model_name = model["model_name"]
    model_config = model["model_config"]

    # Unpack Folds
    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation").get("n_cross_val_folds")

    # Add shared nodes
    pipelines = []
    pipelines.append(create_shared_pipeline())

    # Generate pipeline for the model
    pipelines.append(create_model_pipeline(model_name, model_config["num_shards"], n_cross_val_folds))

    return sum(pipelines)
