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
def process_input_filepaths(
    input_paths: list[InputPathsMultiFold],
) -> list[dict]:
    """Function to create input matrices dataset."""
    # Return initialized dataclass objects as dictionaries
    return [asdict(v) for v in input_paths]


def combine_matrix_pairs(
    input_matrices: dict[str, any],
    available_ground_truth_cols: list[str],
    perform_multifold: bool,
    assert_data_consistency: bool,
) -> tuple[pl.LazyFrame, dict[str, any]]:
    """Function to combine matrix pairs for all folds and models.

    Performs data consistency check or matrix harmonization (restrict to common elements for consistent evaluation) as requested.

    Args:
        input_matrices: Dictionary containing predictions for all models and folds.
        available_ground_truth_cols: List of available ground truth columns.
        perform_multifold: Whether to perform multifold uncertainty estimation. If False, only the first fold of data is used.
        assert_data_consistency: Whether to assert data consistency. If False, matrix harmonization is performed.

    Returns:
        Tuple containing combined matrix pairs dataframe and additional predictions info.
    """

    # Determine number of folds and check consistency across models
    if perform_multifold:
        num_folds = len(next(iter(input_matrices.values()))["predictions_list"])
        if not all(len(model_data["predictions_list"]) == num_folds for model_data in input_matrices.values()):
            raise ValueError("All models must have the same number of folds.")
    else:
        num_folds = 1
    logger.info(f"Number of folds: {num_folds}")

    # Create matrix pairs objects (sparse representations) for each model and fold
    matrix_pairs_dict = {
        (model_name, fold): give_matrix_pairs_from_lazyframe(
            model_data["predictions_list"][fold], available_ground_truth_cols
        )
        for model_name, model_data in input_matrices.items()
        for fold in range(num_folds)
    }
    logger.info(f"MatrixPairs dictionary generated.")

    # Check drugs and diseases list consistency across folds for each model
    is_base_consistent = all(
        check_base_matrices_consistent(*[matrix_pairs_dict[(model_name, fold)] for fold in range(num_folds)])
        for model_name in input_matrices.keys()
    )
    if not is_base_consistent:
        raise ValueError("Drug and disease lists are not consistent across folds.")

    logger.info(f"Drug and disease lists are consistent across folds for all models.")

    # For each fold, check data consistency if requested, or else harmonize matrices across models
    combined_pairs_dict = {}  # Dictionary of MatrixPairs objects containing combined matrix pairs for each fold
    for fold in range(num_folds):
        matrix_pairs_for_fold = [matrix_pairs_dict[(model_name, fold)] for model_name in input_matrices.keys()]
        if assert_data_consistency:
            if not check_matrix_pairs_equal(*matrix_pairs_for_fold):
                raise ValueError("assert_data_consistency is True and input data is not consistent across models.")
            else:  # Take any matrix pairs as they are all equal
                combined_pairs_dict[fold] = matrix_pairs_for_fold[0]

        else:
            combined_pairs_dict[fold] = harmonize_matrix_pairs(*matrix_pairs_for_fold)

    logger.info(f"Combined matrix pairs dictionary generated.")

    # Convert MatrixPairs objects to LazyFrames with string values for fold indices
    combined_pairs_df_dict = {
        "fold_" + str(fold): combined_pairs_dict[fold].to_lazyframe() for fold in range(num_folds)
    }

    # Generate additional information about the predictions
    predictions_info = {
        "model_names": list(input_matrices.keys()),
        "num_folds": num_folds,
        "available_ground_truth_cols": available_ground_truth_cols,
    }

    return combined_pairs_df_dict, predictions_info


def restrict_predictions(
    input_matrices: dict[str, any],
    combined_pairs_df_dict: dict[str, pl.LazyFrame],
    predictions_info: dict[str, any],
) -> dict[str, pl.LazyFrame]:
    """Function to restrict predictions to the common elements of the matrix pairs for all folds and models.

    Args:
        input_matrices: Dictionary containing predictions for all models and folds.
        combined_pairs_df_dict: Dictionary of LazyFrames containing combined matrix pairs for all folds.

    Returns:
        Dictionary containing restricted predictions for all models and folds.
    """
    return {
        model_name + "_fold_" + str(fold): (
            combined_pairs_df_dict["fold_" + str(fold)].join(
                input_matrices[model_name]["predictions_list"][fold].select(
                    "source", "target", pl.col(input_matrices[model_name]["score_col_name"]).alias("score")
                ),
                on=["source", "target"],
                how="left",
            )
        )
        for model_name in predictions_info["model_names"]
        for fold in range(predictions_info["num_folds"])
    }


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
