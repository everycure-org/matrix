import logging
from dataclasses import asdict
from functools import reduce

import matplotlib.pyplot as plt
import polars as pl
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .input_paths import InputPathsMultiFold
from .matrix_pairs import (
    MatrixPairs,
    check_base_matrices_consistent,
    check_matrix_pairs_equal,
    give_matrix_pairs_from_lazyframe,
    harmonize_matrix_pairs,
)

logger = logging.getLogger(__name__)


@inject_object()
def create_input_matrices_dataset(
    input_paths: list[InputPathsMultiFold],
) -> list[dict]:
    """Function to create input matrices dataset."""
    # Return initialized dataclass objects as dictionaries
    return [asdict(v) for v in input_paths]


def _combine_matrix_pairs_for_all_folds(
    matrix_pairs_list: list[MatrixPairs],
    available_ground_truth_cols: list[str],
    num_folds: int,
) -> pl.LazyFrame:
    """Function to combine MatrixPairs objects for all folds into a single LazyFrame.

    Args:
        matrix_pairs_list: List of MatrixPairs objects for each fold.
        available_ground_truth_cols: List of available ground truth columns.
        num_folds: Number of folds.

    Returns:
        LazyFrame containing combined matrix pairs for all folds.
        Ground truth columns are renamed to include fold information.
    """
    matrix_pairs_df_list = [
        matrix_pairs_list[fold]
        .to_lazyframe()
        .rename({col: f"{col}_fold_{fold}" for col in available_ground_truth_cols})
        for fold in range(num_folds)
    ]
    return reduce(lambda x, y: x.join(y, on=["source", "target"], how="left"), matrix_pairs_df_list)


def _join_scores_to_combined_matrix_pairs(
    combined_matrix_pairs: pl.LazyFrame,
    input_matrices: dict[str, any],
    num_folds: int,
) -> pl.LazyFrame:
    """Function to join scores to combined matrix pairs.

    Args:
        combined_matrix_pairs: LazyFrame containing combined matrix pairs for all folds.
        input_matrices: Dictionary containing predictions for all models and folds.
        num_folds: Number of folds.

    Returns:
        LazyFrame containing combined predictions with scores for models and folds joined.
    """
    for model_name, model_data in input_matrices.items():
        for fold in range(num_folds):
            lazy_matrix = model_data["predictions_list"][fold]
            score_col_name = model_data["score_col_name"]
            combined_matrix_pairs = combined_matrix_pairs.join(
                lazy_matrix.select("source", "target", pl.col(score_col_name).alias(f"score_{model_name}_fold_{fold}")),
                on=["source", "target"],
                how="left",
            )
    return combined_matrix_pairs


def combine_predictions(
    input_matrices: dict[str, any],
    available_ground_truth_cols: list[str],
    perform_multifold: bool,
    assert_data_consistency: bool,
) -> tuple[pl.LazyFrame, dict[str, any]]:
    """Function to combine predictions.

    Performs data consistency check or matrix harmonization (restrict to common elements for consistent evaluation) as requested.

    Args:
        input_matrices: Dictionary containing predictions for all models and folds.
        available_ground_truth_cols: List of available ground truth columns.
        perform_multifold: Whether to perform multifold uncertainty estimation. If False, only the first fold of data is used.
        assert_data_consistency: Whether to assert data consistency. If False, matrix harmonization is performed.

    Returns:
        Tuple containing combined predictions and additional predictions info.
    """

    # Determine number of folds and check consistency across models
    if perform_multifold:
        num_folds = len(next(iter(input_matrices.values()))["predictions_list"])
        if not all(len(model_data["predictions_list"]) == num_folds for model_data in input_matrices.values()):
            raise ValueError("All models must have the same number of folds.")
    else:
        num_folds = 1

    # Create matrix pairs objects (sparse representations) for each model and fold
    matrix_pairs_dict = {
        (model_name, fold): give_matrix_pairs_from_lazyframe(lazy_df, available_ground_truth_cols)
        for model_name, model_data in input_matrices.items()
        for fold, lazy_df in enumerate(model_data["predictions_list"])
    }

    # Check drugs and diseases list consistency across folds for each model
    is_base_consistent = all(
        check_base_matrices_consistent(*[matrix_pairs_dict[(model_name, fold)] for fold in range(num_folds)])
        for model_name in input_matrices.keys()
    )
    if not is_base_consistent:
        raise ValueError("Drug and disease lists are not consistent across folds.")

    # For each fold, check data consistency if requested, or else harmonize matrices across models
    combined_matrix_pairs_list = []  # List of harmonized MatrixPairs objects for each fold
    for fold in range(num_folds):
        matrix_pairs_for_fold = [matrix_pairs_dict[(model_name, fold)] for model_name in input_matrices.keys()]
        if assert_data_consistency:
            if not check_matrix_pairs_equal(*matrix_pairs_for_fold):
                raise ValueError("assert_data_consistency is True and input data is not consistent across models.")
            else:
                combined_matrix_pairs_list.append(
                    matrix_pairs_for_fold[0]
                )  # Take any matrix pairs as they are all equal

        else:
            combined_matrix_pairs_list.append(harmonize_matrix_pairs(*matrix_pairs_for_fold))

    # Combine matrix pairs for all folds into a single LazyFrame
    combined_matrix_pairs = _combine_matrix_pairs_for_all_folds(
        combined_matrix_pairs_list, available_ground_truth_cols, num_folds
    )

    # Join score columns to combined matrix
    combined_predictions = _join_scores_to_combined_matrix_pairs(combined_matrix_pairs, input_matrices, num_folds)

    # Generate additional information about the predictions
    predictions_info = {
        "model_names": list(input_matrices.keys()),
        "num_folds": num_folds,
        "available_ground_truth_cols": available_ground_truth_cols,
    }

    return combined_predictions, predictions_info


@inject_object()
def run_evaluation(
    perform_multifold: bool,
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    combined_predictions: pl.LazyFrame,
    predictions_info: dict[str, any],
) -> pl.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")

    if perform_multifold:
        if perform_bootstrap:
            return evaluation.evaluate_bootstrap_multi_fold(combined_predictions, predictions_info)
        else:
            return evaluation.evaluate_multi_fold(combined_predictions, predictions_info)
    else:
        if perform_bootstrap:
            return evaluation.evaluate_bootstrap_single_fold(combined_predictions, predictions_info)
        else:
            return evaluation.evaluate_single_fold(combined_predictions, predictions_info)


@inject_object()
def plot_results(
    perform_multifold: bool,
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    results: pl.DataFrame,
    combined_predictions: pl.LazyFrame,
    predictions_info: dict[str, any],
) -> plt.Figure:
    """Function to plot results."""
    is_plot_errors = perform_multifold or perform_bootstrap
    return evaluation.plot_results(results, combined_predictions, predictions_info, is_plot_errors)
