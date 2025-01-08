from typing import List, Union

from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.pipelines.modelling import nodes as modelling_nodes
from matrix.kedro4argo_node import ArgoNode

from . import nodes


def _create_evaluation_fold_pipeline(model: str, evaluation: str, fold: Union[str, int]) -> Pipeline:
    """Create pipeline for single model, evaluation and fold.

    Args:
        model: model name
        evaluation: name of evaluation suite to generate
        fold: fold to generate
    Returns:
        Pipeline with nodes for given model, evaluation and fold
    """
    return pipeline(
        [
            ArgoNode(
                func=nodes.generate_test_dataset,
                inputs=[
                    f"matrix_generation.{model}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                    f"modelling.model_input.fold_{fold}.splits",
                    "params:matrix_generation.treat_score_col_name",
                ],
                outputs=f"evaluation.{model}.fold_{fold}.{evaluation}.model_output.pairs",
                name=f"create_{model}_{evaluation}_evaluation_pairs_fold_{fold}",
            ),
            ArgoNode(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.{model}.fold_{fold}.{evaluation}.model_output.pairs",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{model}.fold_{fold}.{evaluation}.reporting.result",
                name=f"create_{model}_{evaluation}_evaluation_fold_{fold}",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{model}.{evaluation}.fold_{fold}"],
    )


def create_model_pipeline(
    model: str, evaluation_names: str, folds_lst: List[Union[str, int]], n_splits: int
) -> Pipeline:
    """Create pipeline to evaluate a single model.

    Args:
        model: model name
        num_shards: number of shard to generate
        folds_lst: lists of folds (e.g. [0, 1, 2, 3, "full"] if n_splits=3)
        n_splits: number of splits
    Returns:
        Pipelines with evaluation nodes for given model
    """

    pipelines = []

    # Evaluate each fold
    for fold in folds_lst:
        for evaluation in evaluation_names:
            pipelines.append(
                pipeline(
                    _create_evaluation_fold_pipeline(model, evaluation, fold),
                    tags=[model, evaluation],
                )
            )

    # Consolidate all results
    for evaluation in evaluation_names:
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=modelling_nodes.aggregate_metrics,
                        inputs=[
                            "params:modelling.aggregation_functions",
                            *[
                                f"evaluation.{model}.fold_{fold}.{evaluation}.reporting.result"
                                for fold in range(n_splits)
                            ],
                        ],
                        outputs=f"evaluation.{model}.{evaluation}.reporting.result_aggregated",
                        name=f"aggregate_{model}_{evaluation}_evaluation_results",
                        tags=[model, evaluation],
                    ),
                    # Reduce the aggregate results for simpler readout in MLFlow (e.g. only report mean)
                    ArgoNode(
                        func=nodes.reduce_aggregated_results,
                        inputs=[
                            f"evaluation.{model}.{evaluation}.reporting.result_aggregated",
                            "params:evaluation.reported_aggregations",
                        ],
                        outputs=f"evaluation.{model}.{evaluation}.reporting.result_aggregated_reduced",
                        name=f"reduce_aggregated_{model}_{evaluation}_evaluation_results",
                        tags=[model, evaluation],
                    ),
                ]
            )
        )

    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    """Create evaluation pipeline.

    Pipeline is created dynamically, based on the following dimensions:
        - Models, i.e., type of model, e.g. random forst
        - Folds, i.e., number of folds to train/evaluation
        - Evaluations, i.e., type evaluation suite to run
    """

    # Unpack params
    models = settings.DYNAMIC_PIPELINES_MAPPING.get("modelling")

    # Unpack folds
    n_splits = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_splits")
    folds_lst = list(range(n_splits))

    # Unpack evaluation names
    evaluations = settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")
    evaluation_names = [ev["evaluation_name"] for ev in evaluations]

    # Generate pipelines for each model
    pipelines = []
    for model in models.keys():
        pipelines.append(create_model_pipeline(model, evaluation_names, folds_lst, n_splits))

    # Consolidate metrics across models and folds
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.consolidate_evaluation_reports,
                    inputs={
                        # Consolidate aggregated reports per model fold
                        **{
                            f"{model}.{evaluation}.fold_{fold}": f"evaluation.{model}.fold_{fold}.{evaluation}.reporting.result"
                            for model in models.keys()
                            for evaluation in evaluation_names
                            for fold in folds_lst
                        },
                        # Consolidate aggregated reports per model
                        **{
                            f"{model}.{evaluation}.aggregated": f"evaluation.{model}.{evaluation}.reporting.result_aggregated"
                            for model in models.keys()
                            for evaluation in evaluation_names
                        },
                    },
                    outputs="evaluation.reporting.master_report",
                    name="consolidate_evaluation_reports",
                )
            ]
        )
    )

    return sum(pipelines)
