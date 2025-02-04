from typing import List, Union

from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ArgoNode
from matrix.pipelines.modelling import nodes as modelling_nodes
from matrix.pipelines.modelling.utils import partial_fold

from . import nodes


def _create_evaluation_fold_pipeline(evaluation: str, fold: Union[str, int]) -> Pipeline:
    """Create pipeline for single model, evaluation and fold.

    Args:
        evaluation: name of evaluation suite to generate
        fold: fold to generate

    Returns:
        Pipeline with nodes for given model, evaluation and fold
    """
    return pipeline(
        [
            ArgoNode(
                func=partial_fold(nodes.generate_test_dataset, fold, arg_name="known_pairs"),
                inputs={
                    "known_pairs": "modelling.model_input.splits",
                    "matrix": f"matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                    "generator": f"params:evaluation.{evaluation}.evaluation_options.generator",
                    "score_col_name": "params:matrix_generation.treat_score_col_name",
                },
                outputs=f"evaluation.fold_{fold}.{evaluation}.model_output.pairs",
                name=f"create_{evaluation}_evaluation_pairs_fold_{fold}",
            ),
            ArgoNode(
                func=nodes.evaluate_test_predictions,
                inputs=[
                    f"evaluation.fold_{fold}.{evaluation}.model_output.pairs",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.fold_{fold}.{evaluation}.reporting.result",
                name=f"create_{evaluation}_evaluation_fold_{fold}",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{evaluation}.fold_{fold}"],
    )


def _create_core_stability_pipeline(fold_main: str, fold_to_compare: str, evaluation: str) -> Pipeline:
    if evaluation != "rank_commonality":
        core_pipeline = pipeline(
            [
                ArgoNode(
                    func=nodes.generate_overlapping_dataset,
                    inputs=[
                        f"params:evaluation.{evaluation}.evaluation_options.generator",
                        f"matrix_generation.fold_{fold_main}.model_output.sorted_matrix_predictions@pandas",
                        f"matrix_generation.fold_{fold_to_compare}.model_output.sorted_matrix_predictions@pandas",
                    ],
                    outputs=f"evaluation.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.pairs@pandas",
                    name=f"create_{fold_main}_{fold_to_compare}_{evaluation}_evaluation_pairs",
                ),
                ArgoNode(
                    func=nodes.evaluate_stability_predictions,
                    inputs=[
                        f"evaluation.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.pairs@pandas",
                        f"params:evaluation.{evaluation}.evaluation_options.stability",
                        f"matrix_generation.fold_{fold_main}.model_output.sorted_matrix_predictions@pandas",
                        f"matrix_generation.fold_{fold_to_compare}.model_output.sorted_matrix_predictions@pandas",
                    ],
                    outputs=f"evaluation.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.result",
                    name=f"calculate_{fold_main}_{fold_to_compare}_{evaluation}",
                ),
            ],
            tags=["stability-metrics"],
        )
    else:
        core_pipeline = pipeline(
            [
                ArgoNode(
                    func=nodes.calculate_rank_commonality,
                    inputs=[
                        f"evaluation.fold_{fold_main}.fold_{fold_to_compare}.stability_ranking.model_stability_output.result",
                        f"evaluation.fold_{fold_main}.fold_{fold_to_compare}.stability_overlap.model_stability_output.result",
                    ],
                    outputs=f"evaluation.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.result",
                    name=f"calculate_{fold_main}_{fold_to_compare}_{evaluation}",
                ),
            ],
            tags=["stability-metrics"],
        )
    return core_pipeline


# def create_model_pipeline(model: str, evaluation_names: List[str], n_cross_val_folds: int) -> Pipeline:
def create_model_pipeline(evaluation_names: List[str], n_cross_val_folds: int) -> Pipeline:
    """Create pipeline to evaluate a single model.

    Args:
        evaluation_names: List of evaluation names.
        n_cross_val_folds: number of folds for cross-validation (i.e. number of test/train splits, not including fold with full training data)

    Returns:
        Pipelines with evaluation nodes for given model
    """

    pipelines = []

    # Evaluate each fold
    for fold in range(n_cross_val_folds):
        for evaluation in evaluation_names:
            pipelines.append(
                pipeline(
                    _create_evaluation_fold_pipeline(evaluation, fold),
                    tags=[evaluation],
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
                                f"evaluation.fold_{fold}.{evaluation}.reporting.result"
                                for fold in range(n_cross_val_folds)
                            ],
                        ],
                        outputs=f"evaluation.{evaluation}.reporting.result_aggregated",
                        name=f"aggregate_{evaluation}_evaluation_results",
                        tags=[evaluation],
                    ),
                    # Reduce the aggregate results for simpler readout in MLFlow (e.g. only report mean)
                    ArgoNode(
                        func=nodes.reduce_aggregated_results,
                        inputs=[
                            f"evaluation.{evaluation}.reporting.result_aggregated",
                            "params:evaluation.reported_aggregations",
                        ],
                        outputs=f"evaluation.{evaluation}.reporting.result_aggregated_reduced",
                        name=f"reduce_aggregated_{evaluation}_evaluation_results",
                        tags=[evaluation],
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

    # Unpack number of splits
    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING.get("cross_validation").get("n_cross_val_folds")

    # Unpack evaluation names
    evaluation_names = [ev["evaluation_name"] for ev in settings.DYNAMIC_PIPELINES_MAPPING.get("evaluation")]

    # Generate pipelines for each model
    pipelines = []
    pipelines.append(create_model_pipeline(evaluation_names, n_cross_val_folds))

    # Consolidate metrics across models and folds
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.consolidate_evaluation_reports,
                    inputs={
                        # Consolidate aggregated reports per model fold
                        **{
                            f"{evaluation}.fold_{fold}": f"evaluation.fold_{fold}.{evaluation}.reporting.result"
                            for evaluation in evaluation_names
                            for fold in range(n_cross_val_folds)
                        },
                        # Consolidate aggregated reports per model
                        **{
                            f"{evaluation}.aggregated": f"evaluation.{evaluation}.reporting.result_aggregated"
                            for evaluation in evaluation_names
                        },
                    },
                    outputs="evaluation.reporting.master_report",
                    name="consolidate_evaluation_reports",
                )
            ]
        )
    )
    # Calculate stability between folds
    for stability in settings.DYNAMIC_PIPELINES_MAPPING.get("stability"):
        for fold_main in range(n_cross_val_folds + 1):
            for fold_to_compare in range(
                n_cross_val_folds + 1
            ):  # If we dont want to inclue the full training data, remove +1
                if fold_main == fold_to_compare:
                    continue
                pipelines.append(
                    pipeline(
                        _create_core_stability_pipeline(fold_main, fold_to_compare, stability["stability_name"]),
                    )
                )

    return sum(pipelines)
