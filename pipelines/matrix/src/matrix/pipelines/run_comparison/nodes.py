import logging
from collections.abc import Callable
from functools import reduce

import bracex
import matplotlib.pyplot as plt
import polars as pl
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .matrix_pairs import MatrixPairs, give_matrix_pairs_from_lazyframe

logger = logging.getLogger(__name__)


def process_input_filepaths(
    input_paths: list[dict],
) -> list[dict]:
    """Function to create input matrices dataset.

    This node will be outputted as a custom MultiPredictionsDataset which handles lazy loading of the predictions.

    Args:
        List containing dictionaries with keys:
        name (str), file_paths_list (list[str]), file_format (str), score_col_name (str)
    """
    # Check uniqueness of names
    names = [model["name"] for model in input_paths]
    if len(set(names)) != len(names):
        raise ValueError("All model names must be unique.")

    # Perform brace expansion
    for idx in range(len(input_paths)):
        expanded_paths_list = []
        for path in input_paths[idx]["fold_paths_list"]:
            expanded_paths_list.extend(bracex.expand(path))
        input_paths[idx]["fold_paths_list"] = expanded_paths_list

    # Check values of file-formats
    file_formats = {model["file_format"] for model in input_paths}
    allowed_formats = ["csv", "parquet"]
    if not file_formats.issubset(set(allowed_formats)):
        raise ValueError("File format must be one of the following:", str(allowed_formats))

    return input_paths


def harmonize_matrix_pairs(*matrix_pairs_all: MatrixPairs, **kwargs) -> MatrixPairs:
    """Harmonize a list of MatrixPairs objects.

    Args:
        *matrix_pairs_all: List of MatrixPairs objects to harmonize.
    """
    return reduce(lambda x, y: x.harmonize(y, **kwargs), matrix_pairs_all)


def check_base_matrices_consistent(
    *matrix_pairs_all: MatrixPairs,
) -> bool:
    """Check if a list of MatrixPairs objects have the same drugs and diseases lists."""
    return all(x.is_same_base_matrix(y) for x in matrix_pairs_all for y in matrix_pairs_all[1:])


def check_matrix_pairs_equal(
    *matrix_pairs_all: MatrixPairs,
) -> bool:
    """Check if a list of MatrixPairs objects are equal.

    Args:
        *matrix_pairs_all: List of MatrixPairs objects to check.
    """
    return all(x == matrix_pairs_all[0] for x in matrix_pairs_all[1:])


def combine_matrix_pairs(
    input_matrices: dict[str, any],
    available_ground_truth_cols: list[str],
    apply_harmonization: bool,
) -> tuple[pl.LazyFrame, dict[str, any]]:
    """Function to combine matrix pairs for all folds and models.

    Performs data consistency check or matrix harmonization (restrict to common elements for consistent evaluation) as requested.

    Args:
        input_matrices: Dictionary containing predictions for all models and folds.
        available_ground_truth_cols: List of available ground truth columns.
        apply_harmonization: Whether to perform data harmonization. If False, data consistency is asserted.

    Returns:
        Tuple containing combined matrix pairs dataframe (without scores) and additional predictions info.
    """
    # Determine number of folds and check consistency across models
    num_folds = len(next(iter(input_matrices.values()))["predictions_list"])
    if not all(len(model_data["predictions_list"]) == num_folds for model_data in input_matrices.values()):
        raise ValueError("All models must have the same number of folds.")
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

    # For each fold, harmonize matrices across models if requested, else check data consistency
    combined_pairs_dict = {}  # Dictionary of MatrixPairs objects containing combined matrix pairs for each fold
    for fold in range(num_folds):
        matrix_pairs_for_fold = [matrix_pairs_dict[(model_name, fold)] for model_name in input_matrices.keys()]
        if apply_harmonization:
            combined_pairs_dict[fold] = harmonize_matrix_pairs(*matrix_pairs_for_fold)
        else:
            if not check_matrix_pairs_equal(*matrix_pairs_for_fold):
                raise ValueError("apply_harmonization is False and input data is not consistent across models.")
            else:  # Take any matrix pairs as they are all equal
                combined_pairs_dict[fold] = matrix_pairs_for_fold[0]

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
    combined_pairs: dict[str, Callable[[], pl.LazyFrame]],
    predictions_info: dict[str, any],
) -> dict[str, pl.LazyFrame]:
    """Function to restrict predictions to the common elements of the matrix pairs for all folds and models.

    Args:
        input_matrices: Dictionary containing predictions for all models and folds.
        combined_pairs: Dictionary of PartitionedDataset load fn's returning combined matrix pairs for each fold

    Returns:
        Dictionary containing restricted predictions for all models and folds.
    """
    return {
        model_name + "_fold_" + str(fold): (
            combined_pairs["fold_" + str(fold)]().join(
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
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    combined_predictions: dict[str, Callable[[], pl.LazyFrame]],  # Dictionary of PartitionedDataset load fn's
    predictions_info: dict[str, any],
) -> pl.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")
    return evaluation.evaluate(combined_predictions, predictions_info, perform_bootstrap)


@inject_object()
def plot_results(
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    results: pl.DataFrame,
    combined_pairs: dict[str, Callable[[], pl.LazyFrame]],  # Dictionary of PartitionedDataset load fn's
    predictions_info: dict[str, any],
) -> plt.Figure:
    """Function to plot results."""
    return evaluation.plot_results(results, combined_pairs, predictions_info, perform_bootstrap)
