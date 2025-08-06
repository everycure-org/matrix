from functools import partial
from typing import List, Union

from kedro.pipeline import Pipeline, pipeline
from matrix import settings
from matrix.kedro4argo_node import ARGO_NODE_MEDIUM_MATRIX_GENERATION, ArgoNode
from matrix.pipelines.modelling.utils import partial_fold

from . import nodes


def _create_evaluation_fold_pipeline(
    evaluation: str, fold: Union[str, int], matrix_input: str, score_col_name: str
) -> Pipeline:
    """Create pipeline for single model, evaluation and fold.

    Args:
        evaluation: name of evaluation suite to generate
        fold: fold to generate
        matrix_input: name of the matrix input
        score_col_name: name of the score column to use

    Returns:
        Pipeline with nodes for given model, evaluation and fold
    """
    return pipeline(
        [
            ArgoNode(
                func=partial_fold(
                    partial(nodes.generate_test_dataset, score_col_name=score_col_name), fold, arg_name="known_pairs"
                ),
                inputs={
                    "known_pairs": "modelling.model_input.splits@pandas",
                    "matrix": f"{matrix_input}.fold_{fold}.model_output.sorted_matrix_predictions@pandas",
                    "generator": f"params:evaluation.{evaluation}.evaluation_options.generator",
                },
                outputs=f"evaluation.{matrix_input}.fold_{fold}.{evaluation}.model_output.pairs",
                name=f"{matrix_input}.create_{evaluation}_evaluation_pairs_fold_{fold}",
            ),
            ArgoNode(
                func=partial(nodes.evaluate_test_predictions, score_col_name=score_col_name),
                inputs=[
                    f"evaluation.{matrix_input}.fold_{fold}.{evaluation}.model_output.pairs",
                    f"params:evaluation.{evaluation}.evaluation_options.evaluation",
                ],
                outputs=f"evaluation.{matrix_input}.fold_{fold}.{evaluation}.reporting.result",
                name=f"{matrix_input}.create_{evaluation}_evaluation_fold_{fold}",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{matrix_input}.{evaluation}.fold_{fold}"],
    )


def _create_core_stability_pipeline(
    fold_main: str, fold_to_compare: str, evaluation: str, matrix_input: str, score_col_name: str
) -> Pipeline:
    if evaluation != "rank_commonality":
        pipeline_nodes = [
            ArgoNode(
                func=partial(nodes.generate_overlapping_dataset, score_col_name=score_col_name),
                inputs=[
                    f"params:evaluation.{evaluation}.evaluation_options.generator",
                    f"{matrix_input}.fold_{fold_main}.model_output.sorted_matrix_predictions@pandas",
                    f"{matrix_input}.fold_{fold_to_compare}.model_output.sorted_matrix_predictions@pandas",
                ],
                outputs=f"evaluation.{matrix_input}.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.pairs@pandas",
                name=f"{matrix_input}.create_{fold_main}_{fold_to_compare}_{evaluation}_evaluation_pairs",
            ),
            ArgoNode(
                func=partial(nodes.evaluate_stability_predictions, score_col_name=score_col_name),
                inputs=[
                    f"evaluation.{matrix_input}.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.pairs@pandas",
                    f"params:evaluation.{evaluation}.evaluation_options.stability",
                    f"{matrix_input}.fold_{fold_main}.model_output.sorted_matrix_predictions@pandas",
                    f"{matrix_input}.fold_{fold_to_compare}.model_output.sorted_matrix_predictions@pandas",
                ],
                outputs=f"evaluation.{matrix_input}.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.result",
                name=f"{matrix_input}.calculate_{fold_main}_{fold_to_compare}_{evaluation}",
            ),
        ]
    else:
        pipeline_nodes = [
            ArgoNode(
                func=nodes.calculate_rank_commonality,
                inputs=[
                    f"evaluation.{matrix_input}.fold_{fold_main}.fold_{fold_to_compare}.stability_ranking.model_stability_output.result",
                    f"evaluation.{matrix_input}.fold_{fold_main}.fold_{fold_to_compare}.stability_overlap.model_stability_output.result",
                ],
                outputs=f"evaluation.{matrix_input}.fold_{fold_main}.fold_{fold_to_compare}.{evaluation}.model_stability_output.result",
                name=f"{matrix_input}.calculate_{fold_main}_{fold_to_compare}_{evaluation}",
                argo_config=ARGO_NODE_MEDIUM_MATRIX_GENERATION,
            ),
        ]
    return pipeline(pipeline_nodes, tags=["stability-metrics"])


def foo(x):
    x.show()


# def create_model_pipeline(model: str, evaluation_names: List[str], n_cross_val_folds: int) -> Pipeline:
def create_model_pipeline(
    evaluation_names: List[str], n_cross_val_folds: int, matrix_input: str, score_col_name: str
) -> Pipeline:
    """Create pipeline to evaluate a single model.

    Args:
        evaluation_names: List of evaluation names.
        n_cross_val_folds: number of folds for cross-validation (i.e. number of test/train splits, not including fold with full training data)

    Returns:
        Pipelines with evaluation nodes for given model
    """

    pipelines = []

    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=foo,
                    inputs=["evaluation.orchard.feedback_data@spark"],
                    name="load_orchard_feedback_data",
                    outputs=None,
                )
            ]
        )
    )

    # Evaluate each fold
    for fold in range(n_cross_val_folds):
        for evaluation in evaluation_names:
            pipelines.append(
                pipeline(
                    _create_evaluation_fold_pipeline(evaluation, fold, matrix_input, score_col_name),
                    tags=[evaluation],
                )
            )

    # Consolidate all results
    for evaluation in evaluation_names:
        pipelines.append(
            pipeline(
                [
                    ArgoNode(
                        func=nodes.aggregate_metrics,
                        inputs=[
                            "params:modelling.aggregation_functions",
                            *[
                                f"evaluation.{matrix_input}.fold_{fold}.{evaluation}.reporting.result"
                                for fold in range(n_cross_val_folds)
                            ],
                        ],
                        outputs=f"evaluation.{matrix_input}.{evaluation}.reporting.result_aggregated",
                        name=f"{matrix_input}.aggregate_{evaluation}_evaluation_results",
                        tags=[evaluation],
                    ),
                    # Reduce the aggregate results for simpler readout in MLFlow (e.g. only report mean)
                    ArgoNode(
                        func=nodes.reduce_aggregated_results,
                        inputs=[
                            f"evaluation.{matrix_input}.{evaluation}.reporting.result_aggregated",
                            "params:evaluation.reported_aggregations",
                        ],
                        outputs=f"evaluation.{matrix_input}.{evaluation}.reporting.result_aggregated_reduced",
                        name=f"{matrix_input}.reduce_aggregated_{evaluation}_evaluation_results",
                        tags=[evaluation],
                    ),
                ]
            )
        )
    return sum(pipelines)


def create_pipeline(matrix_input: str, score_col_name: str) -> Pipeline:
    """Create evaluation pipeline.

    Pipeline is created dynamically, based on the following dimensions:
        - Folds, i.e., number of folds to train/evaluate
        - Evaluations, i.e., type evaluation suite to run
    """

    # Unpack number of splits
    n_cross_val_folds = settings.DYNAMIC_PIPELINES_MAPPING().get("cross_validation").get("n_cross_val_folds")

    # Unpack evaluation names
    evaluation_names = [ev["evaluation_name"] for ev in settings.DYNAMIC_PIPELINES_MAPPING().get("evaluation")]

    # Generate pipelines for each model
    pipelines = []
    pipelines.append(
        create_model_pipeline(evaluation_names, n_cross_val_folds, matrix_input, score_col_name=score_col_name)
    )

    # Consolidate metrics across models and folds
    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.consolidate_evaluation_reports,
                    inputs={
                        # Consolidate aggregated reports per model fold
                        **{
                            f"{evaluation}.fold_{fold}": f"evaluation.{matrix_input}.fold_{fold}.{evaluation}.reporting.result"
                            for evaluation in evaluation_names
                            for fold in range(n_cross_val_folds)
                        },
                        # Consolidate aggregated reports per model
                        **{
                            f"{evaluation}.aggregated": f"evaluation.{matrix_input}.{evaluation}.reporting.result_aggregated"
                            for evaluation in evaluation_names
                        },
                    },
                    outputs=f"evaluation.{matrix_input}.reporting.master_report",
                    name=f"{matrix_input}.consolidate_evaluation_reports",
                )
            ]
        )
    )

    # Calculate stability between folds
    for stability in settings.DYNAMIC_PIPELINES_MAPPING().get("stability"):
        for fold_main in range(n_cross_val_folds + 1):
            for fold_to_compare in range(
                n_cross_val_folds + 1
            ):  # If we dont want to inclue the full training data, remove +1
                if fold_main == fold_to_compare:
                    continue
                pipelines.append(
                    pipeline(
                        _create_core_stability_pipeline(
                            fold_main,
                            fold_to_compare,
                            stability["stability_name"],
                            matrix_input=matrix_input,
                            score_col_name=score_col_name,
                        ),
                    )
                )

    return sum(pipelines)
