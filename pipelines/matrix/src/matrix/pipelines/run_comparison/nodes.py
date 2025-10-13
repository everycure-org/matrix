import logging
from dataclasses import asdict

import matplotlib.pyplot as plt
import polars as pl
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .input_paths import InputPathsMultiFold
from .matrix_pairs import (
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

    # Check data consistency if requested, or else harmonize matrices across all models and folds
    if assert_data_consistency:
        if not check_matrix_pairs_equal(*list(matrix_pairs_dict.values())):
            raise ValueError("assert_data_consistency is True and input data is not consistent across models.")
        else:
            combined_matrix_pairs = list(matrix_pairs_dict.values())[0]  # Take any matrix pairs as they are all equal

    else:
        combined_matrix_pairs = harmonize_matrix_pairs(*list(matrix_pairs_dict.values()))

    # Join score columns to combined matrix
    combined_predictions = combined_matrix_pairs.to_lazyframe()
    for model_name, model_data in input_matrices.items():
        for fold, lazy_matrix in enumerate(model_data["predictions_list"]):
            score_col_name = model_data["score_col_name"]
            matrix = lazy_matrix.select("source", "target", score_col_name)
            combined_predictions = combined_predictions.join(
                matrix.rename({score_col_name: f"score_{model_name}_fold_{fold}"}), on=["source", "target"], how="left"
            )

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
