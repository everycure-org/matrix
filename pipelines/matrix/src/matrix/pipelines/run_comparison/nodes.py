import logging
from dataclasses import asdict

import matplotlib.pyplot as plt
import polars as pl
from matrix_inject.inject import inject_object

from .evaluations import ComparisonEvaluation
from .input_paths import InputPathsMultiFold

logger = logging.getLogger(__name__)


@inject_object()
def create_input_matrices_dataset(
    input_paths: list[InputPathsMultiFold],
) -> list[dict]:
    """Function to create input matrices dataset."""
    # Return initialized dataclass objects as dictionaries
    return [asdict(v) for v in input_paths]


# def _intersection_pairs_dataframes(
#     *dataframes: pl.DataFrame,
# ) -> pl.DataFrame:
#     """Function to take intersection of several drug-disease pairs dataframes."""
#     return reduce(lambda x, y: x.join(y, on=["source", "target"], how="inner"), dataframes)

# def _union_pairs_dataframes(
#     *dataframes: pl.DataFrame,
# ) -> pl.DataFrame:
#     """Function to take union of several drug-disease pairs dataframes."""
#     return reduce(lambda x, y: pl.concat([x, y]).unique(), dataframes)


def harmonize_matrices(
    input_matrices: dict[str, any],
    available_ground_truth_cols: list[str],
    perform_multifold: bool,
    assert_data_consistency: bool,
) -> tuple[pl.LazyFrame, dict[str, any]]:
    """Function to harmonize matrices."""

    # Determine number of folds and check consistency across models
    if perform_multifold:
        num_folds = len(next(iter(input_matrices.values()))["predictions_list"])
        if not all(len(model_data["predictions_list"]) == num_folds for model_data in input_matrices.values()):
            raise ValueError("All models must have the same number of folds.")
    else:
        num_folds = 1

    # Create MatrixPairs objects list comprehension
    # Check base matrix consistency across folds all(x == values[0] for x in values[1:])
    # Check equality if requested all(x == values[0] for x in values[1:]), set combined lazyframe
    # Harmonize matrices with partial

    # Join score columns to combined matrix
    for model_name, model_data in input_matrices.items():
        for fold, lazy_matrix in enumerate(model_data["predictions_list"]):
            score_col_name = model_data["score_col_name"]
            matrix = lazy_matrix.select("source", "target", score_col_name)
            harmonized_matrices = harmonized_matrices.join(
                matrix.rename({score_col_name: f"score_{model_name}_fold_{fold}"}), on=["source", "target"], how="left"
            )

    predictions_info = {
        "model_names": list(input_matrices.keys()),
        "num_folds": num_folds,
    }

    return harmonized_matrices, predictions_info

    # # Collect characterizing data: drugs lists, diseases lists, test pairs dicts, exclusion pairs dicts
    # # Note: Predictions = Drugs x Diseases - Exclusion pairs
    # drugs_lists_dict = {}
    # diseases_lists_dict = {}
    # exclusion_pairs_dict = {}
    # test_pairs_dict = {}
    # for idx, (model_name, model_data) in enumerate(input_matrices.items()):
    #     for fold, lazy_matrix in enumerate(model_data["predictions_list"]):
    #         # Materialize matrix in memory
    #         matrix = lazy_matrix.select("source", "target", *available_ground_truth_cols).collect()

    #         # Extract drug and disease lists
    #         drugs_lists_dict[(model_name, fold)] = matrix["source"].unique()
    #         diseases_lists_dict[(model_name, fold)] = matrix["target"].unique()

    #         # Extract exclusion pairs
    #         full_matrix = drugs_lists_dict[(model_name, fold)].join(diseases_lists_dict[(model_name, fold)], how="cross")
    #         exclusion_pairs_dict[(model_name, fold)] = full_matrix.join(matrix.select("source", "target"), on=["source", "target"], how="anti")

    #         # Extract test pairs for all ground truth columns
    #         for col_name in available_ground_truth_cols:
    #             test_pairs_dict[(model_name, fold, col_name)] = matrix.filter(col_name).select("source", "target")

    # # Take intersection of drug and disease list over models and folds
    # drugs_list_common = _intersection_pairs_dataframes(*drugs_lists_dict.values())
    # diseases_list_common = _intersection_pairs_dataframes(*diseases_lists_dict.values())

    # # Take union of exclusion pairs over models for each fixed fold
    # exclusion_pairs_common_dict = {
    #     fold : _union_pairs_dataframes(*[exclusion_pairs_dict[(model_name, fold)] for model_name in input_matrices.keys()])
    #     for fold in range(num_folds)
    #     }

    # # Take intersection of test pairs over models for each fixed fold
    # test_pairs_common_dict = {
    #     (col_name, fold) :  _intersection_pairs_dataframes(*[test_pairs_dict[(model_name, col_name, fold)] for model_name in input_matrices.keys()])
    #     for col_name, fold in zip(available_ground_truth_cols, range(num_folds))
    # }

    # # Find common matrix pairs (drugs x diseases - exclusion pairs) and test pairs across all models
    # for idx, (model_name, model_data) in enumerate(input_matrices.items()):
    #     for fold, lazy_matrix in enumerate(model_data["predictions_list"]):
    #         # Materialize matrix in memory
    #         matrix = lazy_matrix.select("source", "target", *available_ground_truth_cols).collect()

    #         # Initialize drugs and diseases lists for first fold and first model
    #         if (idx == 0) and (fold == 0):
    #             drugs_list = matrix["source"].unique()
    #             diseases_list = matrix["target"].unique()
    #         # Take intersection of drugs and diseases for subsequent folds and models, or assert consistency if requested
    #         else:
    #             new_drugs_list = matrix["source"].unique()
    #             new_diseases_list = matrix["target"].unique()
    #             if assert_data_consistency:
    #                 if (new_drugs_list != drugs_list).any():
    #                     raise ValueError("Drugs list is not consistent across models and folds.")
    #                 if (new_diseases_list != diseases_list).any():
    #                     raise ValueError("Diseases list is not consistent across models and folds.")
    #             else:
    #                 drugs_list = new_drugs_list.join(drugs_list, how="inner", on="source")
    #                 diseases_list = new_diseases_list.join(diseases_list, how="inner", on="target")

    #         if idx == 0: # For the first model:
    #             # Initialize test set pairs for the current fold
    #             test_pairs_dict = {
    #                 (col_name, fold) : matrix.filter(col_name).select("source", "target") for col_name in available_ground_truth_cols
    #                 }

    #             # Initialize excluded pairs for the current fold
    #             full_matrix = drugs_list.join(diseases_list, how="cross")
    #             exclusion_pairs = {
    #                 fold: full_matrix.join(matrix.select("source", "target"), on=["source", "target"], how="anti")
    #                 for fold in range(num_folds)
    #                 }
    #         else: # For subsequent models:
    #             # Take intersection of test set pairs for the current fold, or assert consistency if requested
    #             new_test_pairs_dict = {
    #                 (col_name, fold): test_pairs_dict[(col_name, fold)].join(matrix.filter(col_name).select("source", "target"), on=["source", "target"], how="inner")
    #                 for col_name in available_ground_truth_cols
    #                 }
    #             if assert_data_consistency:
    #                 for col_name in available_ground_truth_cols:
    #                     if (new_test_pairs_dict[(col_name, fold)] != test_pairs_dict[(col_name, fold)]).any():
    #                         raise ValueError("Test set pairs are not consistent across models.")
    #             else:
    #                 # Take union of excluded pairs for the current fold
    #                 exclusion_pairs_dict = {fold: full_matrix.join(matrix.select("source", "target"), on=["source", "target"], how="inner") for fold in range(num_folds)}

    #                 test_pairs_dict = new_test_pairs_dict

    #             # Take union of excluded pairs for the current fold, or assert consistency if requested
    #             full_matrix = drugs_list.join(diseases_list, how="cross")
    #             new_exclusion_pairs = {fold: full_matrix.join(matrix.select("source", "target"), on=["source", "target"], how="inner") for fold in range(num_folds)}


@inject_object()
def run_evaluation(
    perform_multifold: bool,
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    harmonized_matrices: pl.LazyFrame,
) -> pl.DataFrame:
    """Function to apply evaluation."""
    logger.info(f"Evaluation is: {evaluation}")

    if perform_multifold:
        if perform_bootstrap:
            return evaluation.evaluate_bootstrap_multi_fold(harmonized_matrices)
        else:
            return evaluation.evaluate_multi_fold(harmonized_matrices)
    else:
        if perform_bootstrap:
            return evaluation.evaluate_bootstrap_single_fold(harmonized_matrices)
        else:
            return evaluation.evaluate_single_fold(harmonized_matrices)


@inject_object()
def plot_results(
    perform_multifold: bool,
    perform_bootstrap: bool,
    evaluation: ComparisonEvaluation,
    results: pl.DataFrame,
    harmonized_matrices: pl.LazyFrame,
) -> plt.Figure:
    """Function to plot results."""
    is_plot_errors = perform_multifold or perform_bootstrap
    return evaluation.plot_results(results, harmonized_matrices, is_plot_errors)
