# # Direct copy-paste from lab-notebooks.
# # These are the plotting functions for the run comparisons that we do manually for every experiment.

# """Functions for plotting curves for model evaluation."""

# from itertools import combinations
# from typing import List, Optional, Tuple

# import matplotlib.pyplot as plt
# import numpy as np
# import polars as pl
# from evaluation import (
#     give_average_commonality,
#     give_average_entropy_at_n,
#     give_average_hit_at_k_bootstraps,
#     give_average_hit_at_k_folds,
#     give_average_precision_recall_curve_bootstraps,
#     give_average_precision_recall_curve_folds,
#     give_average_recall_at_n_bootstraps,
#     give_average_recall_at_n_folds,
#     give_commonality,
#     give_entropy_at_n,
#     give_hit_at_k,
#     give_precision_recall_curve,
#     give_recall_at_n,
# )


# def plot_av_ranking_metrics(
#     matrices_all: Tuple[List[pl.DataFrame] | pl.DataFrame],
#     model_names: Tuple[str],
#     bool_test_col: str = "is_known_positive",
#     score_col: str = "treat score",
#     perform_sort: bool = True,
#     n_min: int = 10,
#     n_max: int = 100000,
#     n_steps: int = 1000,
#     k_max: int = 100,
#     sup_title: Optional[str] = None,
#     is_average_folds: bool = False,
#     is_average_bootstraps: bool = False,
#     N_bootstraps: int = 1000,
#     plot_error_bars: bool = True,
#     force_full_y_axis: bool = True,
#     out_of_matrix_mode: bool = False,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots average recall@n and average hit@k for a list of models across folds.

#     NOTE: This function expects the training set to have been taken out of the matrix dataframes.

#     NOTE: The function will work regardless of whether the matrix dataframes
#     are the same length across folds and models. For the results to be meaningful,
#     the dataframes should represent matrices for consistent lists of drugs and diseases.
#     Some pairs may be missing from the matrices, this is not a problem. Neither is having
#     some additional pairs added for "out-of-matrix" evaluation.

#     Args:
#         matrices_all: A tuple containing
#             - lists of matrix data-frames across folds, one for each model, (if is_average_folds is True) or
#             - a single matrix dataframe.
#             Training set should have been taken out of the matrices.
#         model_names: Tuple of model names
#         bool_test_col: Boolean column in the matrix indicating the known positive test set
#         score_col: Column in the matrix containing the treat scores.
#         perform_sort: Whether to sort the matrix by the treat score,
#             or expect the dataframe to be sorted already.
#         n_min: Minimum n value for the recall@n plot
#         n_max: Maximum n value for the recall@n plot
#         n_steps: Number of steps in the recall@n plot
#         k_max: Maximum k value for the hit@k plot
#         sup_title: Title for the plot
#         is_average_folds: Whether to plot the average over folds or individual model results
#         is_average_bootstraps: Whether to plot the average over bootstraps or individual model results
#             If is_average_bootstraps is True, then is_average_folds must be False.
#         N_bootstraps: Number of bootstraps to compute if is_average_bootstraps is True
#         plot_error_bars: Whether to plot the error bars
#         force_full_y_axis: Whether to force the y-axis to be from 0 to 1 for recall@n and hit@k plots
#         out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
#             In this case, the matrix dataframes must also contain boolean columns "in_matrix".
#         save_path: Path to save the plot to.
#     """
#     if is_average_bootstraps and is_average_folds:
#         raise ValueError("is_average_bootstraps and is_average_folds cannot both be True")

#     # List of n values for recall@n plot
#     n_lst = np.linspace(n_min, n_max, n_steps)
#     n_lst = [int(n) for n in n_lst]

#     # Get the matrix length and number of drugs (minimum among all input matrices)
#     if is_average_folds:
#         matrix_length = min([len(matrix) for matrix_folds in matrices_all for matrix in matrix_folds])
#         number_of_drugs = min(
#             [len(matrix["source"].unique()) for matrix_folds in matrices_all for matrix in matrix_folds]
#         )
#     else:
#         matrix_length = min([len(matrix) for matrix in matrices_all])
#         number_of_drugs = min([len(matrix["source"].unique()) for matrix in matrices_all])

#     # Set up the figure with 2 subplots vertically stacked
#     _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

#     # Plot recall@n vs n
#     for model_name, matrix_folds in zip(model_names, matrices_all):
#         if is_average_folds:
#             av_recall, std_recall = give_average_recall_at_n_folds(
#                 matrix_folds,
#                 n_lst,
#                 bool_test_col=bool_test_col,
#                 score_col=score_col,
#                 perform_sort=perform_sort,
#                 out_of_matrix_mode=out_of_matrix_mode,
#             )
#             ax1.plot(n_lst, av_recall, label=model_name)
#             if plot_error_bars:
#                 ax1.fill_between(n_lst, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#         elif is_average_bootstraps:  # In this case, matrix_folds is a single dataframe
#             av_recall, std_recall = give_average_recall_at_n_bootstraps(
#                 matrix_folds,
#                 n_lst,
#                 N_bootstraps,
#                 bool_test_col=bool_test_col,
#                 score_col=score_col,
#                 perform_sort=perform_sort,
#                 out_of_matrix_mode=out_of_matrix_mode,
#             )
#             ax1.plot(n_lst, av_recall, label=model_name)
#             if plot_error_bars:
#                 ax1.fill_between(n_lst, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#         else:  # In this case, matrix_folds is a single dataframe
#             ax1.plot(
#                 n_lst,
#                 give_recall_at_n(
#                     matrix_folds,
#                     n_lst,
#                     bool_test_col=bool_test_col,
#                     score_col=score_col,
#                     perform_sort=perform_sort,
#                     out_of_matrix_mode=out_of_matrix_mode,
#                 ),
#                 label=model_name,
#             )
#     ax1.plot([0, matrix_length], [0, 1], "k--", label="Random classifier", alpha=0.5)
#     ax1.legend()
#     ax1.set_xlabel("n")
#     if is_average_folds:
#         ax1.set_ylabel("Average recall@n")
#     elif is_average_bootstraps:
#         ax1.set_ylabel("Average recall@n (bootstraps)")
#     else:
#         ax1.set_ylabel("Recall@n")
#     ax1.set_xlim(0, n_max)
#     if force_full_y_axis:
#         ax1.set_ylim(0, 1)
#     if is_average_folds:
#         ax1.set_title("Average full matrix Recall@n vs n")
#     elif is_average_bootstraps:
#         ax1.set_title("Average recall@n (bootstraps) vs n")
#     else:
#         ax1.set_title("Recall@n vs n")
#     ax1.grid(True)

#     # Plot hit@k vs k
#     for model_name, matrix_folds in zip(model_names, matrices_all):
#         if is_average_folds:
#             hit_at_k_dict = give_average_hit_at_k_folds(
#                 matrix_folds, k_max, bool_test_col=bool_test_col, score_col=score_col
#             )
#             ax2.plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#             if plot_error_bars:
#                 ax2.fill_between(
#                     hit_at_k_dict["k"],
#                     hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                     hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                     alpha=0.2,
#                 )
#         elif is_average_bootstraps:
#             hit_at_k_dict = give_average_hit_at_k_bootstraps(
#                 matrix_folds, k_max, N_bootstraps, bool_test_col=bool_test_col, score_col=score_col
#             )
#             ax2.plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#             if plot_error_bars:
#                 ax2.fill_between(
#                     hit_at_k_dict["k"],
#                     hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                     hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                     alpha=0.2,
#                 )
#         else:
#             hit_at_k_dict = give_hit_at_k(matrix_folds, k_max, bool_test_col=bool_test_col, score_col=score_col)
#             ax2.plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k"], label=model_name)
#     ax2.plot([0, number_of_drugs], [0, 1], "k--", label="Random classifier", alpha=0.5)
#     ax2.legend()
#     ax2.set_xlabel("k")
#     if is_average_folds:
#         ax2.set_ylabel("Average hit@k")
#     elif is_average_bootstraps:
#         ax2.set_ylabel("Average hit@k (bootstraps)")
#     else:
#         ax2.set_ylabel("Hit@k")
#     if is_average_folds:
#         ax2.set_title("Average disease-specific Hit@k vs k")
#     elif is_average_bootstraps:
#         ax2.set_title("Average hit@k (bootstraps) vs k")
#     else:
#         ax2.set_title("Disease-specific Hit@k vs k")
#     ax2.set_xlim(0, k_max)
#     if force_full_y_axis:
#         ax2.set_ylim(0, 1)
#     ax2.grid(True)

#     plt.suptitle(sup_title)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_negative_metrics(
#     matrices_all: Tuple[List[pl.DataFrame] | pl.DataFrame],
#     model_names: Tuple[str],
#     bool_pos_col: str = "is_known_positive",
#     bool_neg_col: str = "is_known_negative",
#     score_col: str = "treat score",
#     perform_sort: bool = True,
#     n_min: int = 10,
#     n_max: Optional[int] = 10**6,
#     n_steps: int = 1000,
#     k_max: Optional[int] = 200,
#     sup_title: Optional[str] = None,
#     is_average_folds: bool = False,
#     is_average_bootstraps: bool = False,
#     N_bootstraps: int = 1000,
#     plot_error_bars: bool = True,
#     force_full_y_axis: bool = False,
#     out_of_matrix_mode: bool = False,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots:
#         - average full matrix recall@n for negatives, and average hit@k for a list of model across folds.
#         - average precision-recall curve for positives vs. negatives, and average hit@k for a list of models.


#     NOTE: This function expects the training set to have been taken out of the matrix dataframes.

#     Args:
#         matrices_all: A tuple containing
#             - lists of matrix data-frames across folds, one for each model, if is_average_folds is True
#             - a single matrix dataframe, if is_average_folds is False
#             Training set should have been taken out of the matrices.
#         model_names: Tuple of model names
#         bool_test_col: Boolean column in the matrix indicating the known positive test set
#         score_col: Column in the matrix containing the treat scores.
#         perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
#         n_min: Minimum n value for the recall@n plot
#         n_max: Maximum n value for the recall@n plot
#         n_steps: Number of steps in the recall@n plot
#         k_max: Maximum k value for the hit@k plot
#         sup_title: Title for the plot
#         is_average_folds: Whether to plot the average over folds or individual model results
#         is_average_bootstraps: Whether to plot the average over bootstraps or individual model results
#             If is_average_bootstraps is True, then is_average_folds must be False.
#         N_bootstraps: Number of bootstraps to compute if is_average_bootstraps is True
#         plot_error_bars: Whether to plot the error bars
#         force_full_y_axis: Whether to force the y-axis to be from 0 to 1 for recall@n and hit@k plots
#         out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
#             In this case, the matrix dataframes must also contain boolean columns "in_matrix".
#         save_path: Path to save the plot to.
#     """
#     if is_average_bootstraps and is_average_folds:
#         raise ValueError("is_average_bootstraps and is_average_folds cannot both be True")

#     # List of n values for recall@n plot
#     if is_average_folds:
#         matrix_length = min([len(matrix) for matrix_folds in matrices_all for matrix in matrix_folds])
#     else:
#         matrix_length = min([len(matrix) for matrix in matrices_all])
#     if n_max is None:
#         n_max = matrix_length
#     n_lst = np.linspace(n_min, n_max, n_steps)
#     n_lst = [int(n) for n in n_lst]

#     #  Max k value for hit@k plot (number of drugs)
#     if is_average_folds:
#         number_of_drugs = min(
#             [len(matrix["source"].unique()) for matrix_folds in matrices_all for matrix in matrix_folds]
#         )
#     else:
#         number_of_drugs = min([len(matrix["source"].unique()) for matrix in matrices_all])
#     if k_max is None:
#         k_max = number_of_drugs

#     # Set up the figure with 3 subplots vertically stacked
#     _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

#     # Plot Precision-Recall curve
#     for model_name, matrix_folds in zip(model_names, matrices_all):
#         if is_average_folds:
#             x_vals, precision_recall_curve_heights_av, precision_recall_curve_heights_std = (
#                 give_average_precision_recall_curve_folds(matrix_folds, bool_pos_col, bool_neg_col, score_col)
#             )
#             ax1.plot(x_vals, precision_recall_curve_heights_av, label=model_name)
#             ax1.fill_between(
#                 x_vals,
#                 precision_recall_curve_heights_av - precision_recall_curve_heights_std,
#                 precision_recall_curve_heights_av + precision_recall_curve_heights_std,
#                 alpha=0.2,
#             )
#         elif is_average_bootstraps:  # In this case, matrix_folds is a single dataframe
#             x_vals, precision_recall_curve_heights_av, precision_recall_curve_heights_std = (
#                 give_average_precision_recall_curve_bootstraps(
#                     matrix_folds, N_bootstraps, bool_pos_col, bool_neg_col, score_col
#                 )
#             )
#             ax1.plot(x_vals, precision_recall_curve_heights_av, label=model_name)
#             ax1.fill_between(
#                 x_vals,
#                 precision_recall_curve_heights_av - precision_recall_curve_heights_std,
#                 precision_recall_curve_heights_av + precision_recall_curve_heights_std,
#                 alpha=0.2,
#             )
#         else:  # In this case, matrix_folds is a single dataframe
#             prec, rec = give_precision_recall_curve(matrix_folds, bool_pos_col, bool_neg_col, score_col)
#             ax1.plot(rec, prec, label=model_name)
#     ax1.legend()
#     ax1.set_xlabel("Recall")
#     ax1.set_ylabel("Precision")
#     if is_average_folds:
#         ax1.set_title("Average Precision-Recall curve")
#     elif is_average_bootstraps:
#         ax1.set_title("Average Precision-Recall curve (bootstraps)")
#     else:
#         ax1.set_title("Precision-Recall curve")
#     ax1.grid(True)

#     # Plot recall@n vs n
#     for model_name, matrix_folds in zip(model_names, matrices_all):
#         if is_average_folds:
#             av_recall, std_recall = give_average_recall_at_n_folds(
#                 matrix_folds,
#                 n_lst,
#                 bool_test_col=bool_neg_col,
#                 score_col=score_col,
#                 perform_sort=perform_sort,
#                 out_of_matrix_mode=out_of_matrix_mode,
#             )
#             ax2.plot(n_lst, av_recall, label=model_name)
#             if plot_error_bars:
#                 ax2.fill_between(n_lst, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#         elif is_average_bootstraps:  # In this case, matrix_folds is a single dataframe
#             av_recall, std_recall = give_average_recall_at_n_bootstraps(
#                 matrix_folds,
#                 n_lst,
#                 N_bootstraps,
#                 bool_test_col=bool_neg_col,
#                 score_col=score_col,
#                 out_of_matrix_mode=out_of_matrix_mode,
#             )
#             ax2.plot(n_lst, av_recall, label=model_name)
#             if plot_error_bars:
#                 ax2.fill_between(n_lst, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#         else:  # In this case, matrix_folds is a single dataframe
#             ax2.plot(
#                 n_lst,
#                 give_recall_at_n(
#                     matrix_folds,
#                     n_lst,
#                     bool_test_col=bool_neg_col,
#                     score_col=score_col,
#                     perform_sort=perform_sort,
#                     out_of_matrix_mode=out_of_matrix_mode,
#                 ),
#             )
#     ax2.plot([0, n_max], [0, n_max / matrix_length], "k--", label="Random classifier", alpha=0.5)
#     ax2.set_xlabel("n")
#     if is_average_folds:
#         ax2.set_ylabel("Average recall@n (negatives)")
#     elif is_average_bootstraps:
#         ax2.set_ylabel("Average recall@n (negatives)")
#     else:
#         ax2.set_ylabel("Recall@n (negatives)")
#     if is_average_folds:
#         ax2.set_title("Average full matrix Recall@n vs n (negatives - lower is better)")
#     elif is_average_bootstraps:
#         ax2.set_title("Recall@n with bootstrap uncertainty (negatives - lower is better)")
#     else:
#         ax2.set_title("Recall@n vs n (negatives - lower is better)")
#     if force_full_y_axis:
#         ax2.set_ylim(0, 1)
#     ax2.legend()
#     ax2.grid(True)

#     # Plot hit@k vs k
#     for model_name, matrix_folds in zip(model_names, matrices_all):
#         if is_average_folds:
#             hit_at_k_dict = give_average_hit_at_k_folds(
#                 matrix_folds, k_max, bool_test_col=bool_neg_col, score_col=score_col
#             )
#             ax3.plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#             if plot_error_bars:
#                 ax3.fill_between(
#                     hit_at_k_dict["k"],
#                     hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                     hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                     alpha=0.2,
#                 )
#         elif is_average_bootstraps:  # In this case, matrix_folds is a single dataframe
#             hit_at_k_dict = give_average_hit_at_k_bootstraps(
#                 matrix_folds, k_max, N_bootstraps, bool_test_col=bool_neg_col, score_col=score_col
#             )
#             ax3.plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#             if plot_error_bars:
#                 ax3.fill_between(
#                     hit_at_k_dict["k"],
#                     hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                     hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                     alpha=0.2,
#                 )
#         else:  # In this case, matrix_folds is a single dataframe
#             hit_at_k = give_hit_at_k(matrix_folds, k_max, bool_test_col=bool_neg_col, score_col=score_col)
#             ax3.plot(hit_at_k["k"], hit_at_k["hit_at_k"], label=model_name)
#     ax3.plot([0, k_max], [0, k_max / number_of_drugs], "k--", label="Random classifier", alpha=0.5)
#     ax3.set_xlabel("k")
#     if is_average_folds:
#         ax3.set_ylabel("Average hit@k (negatives)")
#     else:
#         ax3.set_ylabel("Hit@k (negatives)")
#     if is_average_folds:
#         ax3.set_title("Average disease-specific Hit@k vs k (negatives - lower is better)")
#     else:
#         ax3.set_title("Disease-specific Hit@k vs k (negatives - lower is better)")
#     if force_full_y_axis:
#         ax3.set_ylim(0, 1)
#     ax3.legend()
#     ax3.grid(True)
#     plt.suptitle(sup_title)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_av_entropy(
#     matrices_all_folds: Tuple[List[pl.DataFrame]],
#     model_names: Tuple[str],
#     score_col: str = "treat score",
#     perform_sort: bool = True,
#     n_min: int = 10,
#     n_max: int = 100000,
#     n_steps: int = 1000,
#     is_average_folds: bool = True,
#     sup_title: Optional[str] = None,
#     plot_error_bars: bool = True,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots average entropy@n for a list of models across folds.


#     Args:
#         matrices_all_folds: A tuple containing lists of matrix data-frames across folds, one for each model.
#             In the case of is_average = False, this is just a tuple of dataframes.
#         model_names: Tuple of model names
#         score_col: Column in the matrix containing the treat scores.
#         perform_sort: Whether to sort the matrix by the treat score,
#             or expect the dataframe to be sorted already.
#         n_min: Minimum n value for the recall@n plot
#         n_max: Maximum n value for the recall@n plot
#         n_steps: Number of steps in the recall@n plot
#         is_average_folds: Whether to plot the average over folds or individual model results
#         sup_title: Title for the plot
#         plot_error_bars: Whether to plot the error bars
#         save_path: Path to save the plot to.
#     """
#     # List of n values for recall@n plot
#     n_lst = np.linspace(n_min, n_max, n_steps)
#     n_lst = [int(n) for n in n_lst]

#     # Set up the figure with 2 subplots vertically stacked
#     _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

#     # Plot drug-entropy@n  and disease-entropy@n vs n
#     for model_name, matrix_folds in zip(model_names, matrices_all_folds):
#         if is_average_folds:
#             entropy_dict = give_average_entropy_at_n(
#                 matrix_folds, n_lst, score_col=score_col, perform_sort=perform_sort
#             )
#             ax1.plot(n_lst, entropy_dict["drug_entropy_mean"], label=model_name)
#             if plot_error_bars:
#                 ax1.fill_between(
#                     n_lst,
#                     entropy_dict["drug_entropy_mean"] - entropy_dict["drug_entropy_std"],
#                     entropy_dict["drug_entropy_mean"] + entropy_dict["drug_entropy_std"],
#                     alpha=0.2,
#                 )
#             ax2.plot(n_lst, entropy_dict["disease_entropy_mean"], label=model_name)
#             if plot_error_bars:
#                 ax2.fill_between(
#                     n_lst,
#                     entropy_dict["disease_entropy_mean"] - entropy_dict["disease_entropy_std"],
#                     entropy_dict["disease_entropy_mean"] + entropy_dict["disease_entropy_std"],
#                     alpha=0.2,
#                 )
#         else:
#             drug_entropy, disease_entropy = give_entropy_at_n(
#                 matrix_folds, n_lst, score_col=score_col, perform_sort=perform_sort
#             )
#             ax1.plot(n_lst, drug_entropy, label=model_name)
#             ax2.plot(n_lst, disease_entropy, label=model_name)

#     # Axis 1 configuration
#     ax1.legend()
#     ax1.set_xlabel("n")
#     if is_average_folds:
#         ax1.set_ylabel("Average drug-entropy@n")
#     else:
#         ax1.set_ylabel("Drug-entropy@n")
#     ax1.set_xlim(0, n_max)
#     ax1.set_ylim(0, 1)
#     if is_average_folds:
#         ax1.set_title("Average drug-entropy@n vs n")
#     else:
#         ax1.set_title("Drug-entropy@n vs n")
#     ax1.grid(True)

#     # Axis 2 configuration
#     ax2.legend()
#     ax2.set_xlabel("n")
#     if is_average_folds:
#         ax2.set_ylabel("Average disease-entropy@n")
#     else:
#         ax2.set_ylabel("Disease-entropy@n")
#     ax2.set_xlim(0, n_max)
#     ax2.set_ylim(0, 1)
#     if is_average_folds:
#         ax2.set_title("Average disease-entropy@n vs n")
#     else:
#         ax2.set_title("Disease-entropy@n vs n")
#     ax2.grid(True)

#     plt.suptitle(sup_title)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_commonality_between_models(
#     matrices_all_folds: Tuple[List[pl.DataFrame]],
#     model_names: Tuple[str],
#     score_col: str = "treat score",
#     perform_sort: bool = True,
#     k_max: int = 100000,
#     k_steps: int = 1000,
#     sup_title: Optional[str] = None,
#     plot_error_bars: bool = True,
#     is_average_folds: bool = True,
#     is_log_scale: bool = False,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots average commonality@k for a list of models across folds.

#     This can be used for plotting the commonality@k for a single fold,
#     just pass in a lists with a single matrix dataframe for the model(s).


#     Args:
#         matrices_all_folds: A tuple containing lists of matrix data-frames across folds, one for each model.
#         model_names: Tuple of model names
#         bool_test_col: Boolean column in the matrix indicating the known positive test set
#         score_col: Column in the matrix containing the treat scores.
#         perform_sort: Whether to sort the matrix by the treat score,
#             or expect the dataframe to be sorted already.
#         k_max: Maximum k value for the commonality@k plot
#         k_steps: Number of steps in the commonality@k plot
#         sup_title: Title for the plot
#         plot_error_bars: Whether to plot the error bars
#         is_average_folds: Whether to plot the average over folds or individual model results
#         is_log_scale: Whether to plot the commonality@k on a log scale
#         save_path: Path to save the plot to.
#     """
#     k_lst = [int(k) for k in np.linspace(1, k_max + 1, k_steps)]

#     _, ax1 = plt.subplots(1, 1, figsize=(10, 5))

#     for (model_name_1, matrix_folds_1), (model_name_2, matrix_folds_2) in combinations(
#         zip(model_names, matrices_all_folds), 2
#     ):
#         if is_average_folds:
#             commonality_df = give_average_commonality(
#                 matrix_folds_1, matrix_folds_2, k_lst, score_col=score_col, perform_sort=perform_sort
#             )
#             ax1.plot(k_lst, commonality_df["commonality_mean"], label=f"{model_name_1} vs {model_name_2}")
#             if plot_error_bars:
#                 ax1.fill_between(
#                     k_lst,
#                     commonality_df["commonality_mean"] - commonality_df["commonality_std"],
#                     commonality_df["commonality_mean"] + commonality_df["commonality_std"],
#                     alpha=0.2,
#                 )
#         else:
#             commonality_df = give_commonality(
#                 matrix_folds_1, matrix_folds_2, k_lst, score_col=score_col, perform_sort=perform_sort
#             )
#             ax1.plot(commonality_df["k"], commonality_df["commonality@k"], label=f"{model_name_1} vs {model_name_2}")

#     if is_log_scale:
#         ax1.set_yscale("log")
#     ax1.legend()
#     ax1.set_xlabel("k")
#     ax1.set_ylabel("Commonality@k")
#     ax1.grid(True)
#     plt.suptitle(sup_title)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_summary_comparison(
#     matrix_all: Tuple[List[pl.DataFrame] | pl.DataFrame],
#     model_names: Tuple[str],
#     sup_title: str = None,
#     scale: str = 1,
#     pos_col: str = "is_known_positive",
#     neg_col: str = "is_known_negative",
#     off_col: str = "is_off_label",
#     trial_col: str = "trial_sig_better",
#     score_col: str = "treat score",
#     n_min_pos: int = 10,
#     n_max_pos: int = 100000,
#     n_steps_pos: int = 1000,
#     k_max_pos: int = 100,
#     n_min_neg: int = 10,
#     n_max_neg: int = 10**6,
#     n_steps_neg: int = 1000,
#     k_max_neg: int = 200,
#     k_max_commonality: int = 100000,
#     k_steps_commonality: int = 1000,
#     perform_sort: bool = False,
#     is_average_folds: bool = False,
#     is_average_bootstraps: bool = False,
#     N_bootstraps: int = 1000,
#     plot_error_bars: bool = True,
#     force_full_y_axis_pos: bool = True,
#     force_full_y_axis_neg: bool = False,
#     pos_label_names=["Standard", "Off-label", "Significantly successful time-resolved clinical trials"],
#     out_of_matrix_mode: bool = False,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots summary metrics for a list of models.

#     Args:
#         matrix_all: A tuple containing lists of matrix data-frames across folds, one for each model.
#             In the case of is_average = False, this is just a tuple of dataframes.
#         model_names: A list of model names
#         pos_col: The column name for the positive ground truth boolean
#         neg_col: The column name for the negative ground truth boolean
#         off_col: The column name for the off-label ground truth boolean
#         trial_col: The column name for the trial ground truth boolean
#         score_col: The column name for the treat score
#         n_min_pos: Min for positive recall@n plot
#         n_max_pos: Max for positive recall@n plot
#         n_steps_pos: Steps for positive recall@n plot
#         k_max_pos: Max for positive hit@k plot
#         n_min_neg: Min for negative recall@n plot
#         n_max_neg: Max for negative recall@n plot
#         n_steps_neg: Steps for negative recall@n plot
#         k_max_neg: Max for negative hit@k plot
#         k_max_commonality: Max for commonality@k plot
#         k_steps_commonality: Steps for commonality plot
#         perform_sort: Whether to sort the examples by the treat score
#         is_average_folds: Whether to plot the average over folds or individual model results
#         is_average_bootstraps: Whether to plot the average over bootstraps or individual model results
#         N_bootstraps: Number of bootstraps to use for the average recall@n plot
#         plot_error_bars: Whether to plot the error bars
#         force_full_y_axis_pos: Whether to force the y-axis to be full for positive recall@n and Hit@k plots
#         force_full_y_axis_neg: Whether to force the y-axis to be full for negative recall@n and Hit@k plots
#         pos_label_names: The names for the positive ground truth labels
#         out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
#             In this case, the matrix dataframes must also contain boolean columns "in_matrix".
#         save_path: Path to save the plot to.
#     """
#     if is_average_bootstraps and is_average_folds:
#         raise ValueError("is_average_bootstraps and is_average_folds cannot both be True")

#     # Create a 3x4 subplot for summary metrics
#     _, ax = plt.subplots(4, 3, figsize=(scale * 30, scale * 20))

#     # List of n values for recall@n plot
#     n_lst_pos = np.linspace(n_min_pos, n_max_pos, n_steps_pos)
#     n_lst_pos = [int(n) for n in n_lst_pos]

#     # Get the matrix length and number of drugs (minimum among all input matrices)
#     if is_average_folds:
#         matrix_length = min([len(matrix) for matrix_folds in matrix_all for matrix in matrix_folds])
#         number_of_drugs = min(
#             [len(matrix["source"].unique()) for matrix_folds in matrix_all for matrix in matrix_folds]
#         )
#     else:
#         matrix_length = min([len(matrix) for matrix in matrix_all])
#         number_of_drugs = min([len(matrix["source"].unique()) for matrix in matrix_all])

#     # Plot positive recall@n and hit@k

#     for i, bool_test_col in enumerate([pos_col, off_col, trial_col]):
#         # Plot recall@n vs n
#         for model_name, matrix_folds in zip(model_names, matrix_all):
#             if is_average_folds:
#                 av_recall, std_recall = give_average_recall_at_n_folds(
#                     matrix_folds,
#                     n_lst_pos,
#                     bool_test_col=bool_test_col,
#                     score_col=score_col,
#                     perform_sort=perform_sort,
#                     out_of_matrix_mode=out_of_matrix_mode,
#                 )
#                 ax[0, i].plot(n_lst_pos, av_recall, label=model_name)
#                 if plot_error_bars:
#                     ax[0, i].fill_between(n_lst_pos, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#             elif is_average_bootstraps:
#                 av_recall, std_recall = give_average_recall_at_n_bootstraps(
#                     matrix_folds,
#                     n_lst_pos,
#                     N_bootstraps,
#                     bool_test_col=bool_test_col,
#                     score_col=score_col,
#                     perform_sort=perform_sort,
#                     out_of_matrix_mode=out_of_matrix_mode,
#                 )
#                 ax[0, i].plot(n_lst_pos, av_recall, label=model_name)
#                 if plot_error_bars:
#                     ax[0, i].fill_between(n_lst_pos, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#             else:
#                 ax[0, i].plot(
#                     n_lst_pos,
#                     give_recall_at_n(
#                         matrix_folds,
#                         n_lst_pos,
#                         bool_test_col=bool_test_col,
#                         score_col=score_col,
#                         perform_sort=perform_sort,
#                         out_of_matrix_mode=out_of_matrix_mode,
#                     ),
#                     label=model_name,
#                 )
#         ax[0, i].plot([0, matrix_length], [0, 1], "k--", label="Random classifier", alpha=0.5)
#         ax[0, i].legend()
#         ax[0, i].set_xlabel("n")
#         if is_average_folds:
#             ax[0, i].set_ylabel("Average recall@n")
#         elif is_average_bootstraps:
#             ax[0, i].set_ylabel("Average recall@n (with bootstrap uncertainty)")
#         else:
#             ax[0, i].set_ylabel("Recall@n")
#         ax[0, i].set_xlim(0, n_max_pos)
#         if force_full_y_axis_pos:
#             ax[0, i].set_ylim(0, 1)
#         if is_average_folds:
#             ax[0, i].set_title(f"Average full matrix Recall@n vs n ({pos_label_names[i]})")
#         elif is_average_bootstraps:
#             ax[0, i].set_title(f"Recall@n with bootstrap uncertainty ({pos_label_names[i]})")
#         else:
#             ax[0, i].set_title(f"Recall@n vs n ({pos_label_names[i]})")
#         ax[0, i].grid(True)

#         # Plot hit@k vs k
#         for model_name, matrix_folds in zip(model_names, matrix_all):
#             if is_average_folds:
#                 hit_at_k_dict = give_average_hit_at_k_folds(
#                     matrix_folds, k_max_pos, bool_test_col=bool_test_col, score_col=score_col
#                 )
#                 ax[1, i].plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#                 if plot_error_bars:
#                     ax[1, i].fill_between(
#                         hit_at_k_dict["k"],
#                         hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                         hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                         alpha=0.2,
#                     )
#             elif is_average_bootstraps:
#                 hit_at_k_dict = give_average_hit_at_k_bootstraps(
#                     matrix_folds, k_max_pos, N_bootstraps, bool_test_col=bool_test_col, score_col=score_col
#                 )
#                 ax[1, i].plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#                 if plot_error_bars:
#                     ax[1, i].fill_between(
#                         hit_at_k_dict["k"],
#                         hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                         hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                         alpha=0.2,
#                     )
#             else:
#                 hit_at_k_dict = give_hit_at_k(matrix_folds, k_max_pos, bool_test_col=bool_test_col, score_col=score_col)
#                 ax[1, i].plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k"], label=model_name)
#         ax[1, i].plot([0, number_of_drugs], [0, 1], "k--", label="Random classifier", alpha=0.5)
#         ax[1, i].legend()
#         ax[1, i].set_xlabel("k")
#         if is_average_folds:
#             ax[1, i].set_ylabel("Average hit@k")
#         elif is_average_bootstraps:
#             ax[1, i].set_ylabel("Average hit@k (with bootstrap uncertainty)")
#         else:
#             ax[1, i].set_ylabel("Hit@k")
#         if is_average_folds:
#             ax[1, i].set_title(f"Average disease-specific Hit@k vs k ({pos_label_names[i]})")
#         elif is_average_bootstraps:
#             ax[1, i].set_title(f"Hit@k with bootstrap uncertainty ({pos_label_names[i]})")
#         else:
#             ax[1, i].set_title(f"Disease-specific Hit@k vs k ({pos_label_names[i]})")
#         ax[1, i].set_xlim(0, k_max_pos)
#         if force_full_y_axis_pos:
#             ax[1, i].set_ylim(0, 1)
#         ax[1, i].grid(True)

#     # Plot precision-recall negative recall@n and hit@k

#     # List of n values for recall@n plot
#     n_lst_neg = np.linspace(n_min_neg, n_max_neg, n_steps_neg)
#     n_lst_neg = [int(n) for n in n_lst_neg]

#     # Plot Precision-Recall curve
#     for model_name, matrix_folds in zip(model_names, matrix_all):
#         if is_average_folds:
#             x_vals, precision_recall_curve_heights_av, precision_recall_curve_heights_std = (
#                 give_average_precision_recall_curve_folds(matrix_folds, pos_col, neg_col, score_col)
#             )
#             ax[2, 0].plot(x_vals, precision_recall_curve_heights_av, label=model_name)
#             if plot_error_bars:
#                 ax[2, 0].fill_between(
#                     x_vals,
#                     precision_recall_curve_heights_av - precision_recall_curve_heights_std,
#                     precision_recall_curve_heights_av + precision_recall_curve_heights_std,
#                     alpha=0.2,
#                 )
#         elif is_average_bootstraps:
#             x_vals, precision_recall_curve_heights_av, precision_recall_curve_heights_std = (
#                 give_average_precision_recall_curve_bootstraps(
#                     matrix_df=matrix_folds,
#                     N_bootstraps=N_bootstraps,
#                     bool_test_col_pos=pos_col,
#                     bool_test_col_neg=neg_col,
#                     score_col=score_col,
#                 )
#             )
#             ax[2, 0].plot(x_vals, precision_recall_curve_heights_av, label=model_name)
#             if plot_error_bars:
#                 ax[2, 0].fill_between(
#                     x_vals,
#                     precision_recall_curve_heights_av - precision_recall_curve_heights_std,
#                     precision_recall_curve_heights_av + precision_recall_curve_heights_std,
#                     alpha=0.2,
#                 )
#         else:
#             prec, rec = give_precision_recall_curve(matrix_folds, pos_col, neg_col, score_col)
#             ax[2, 0].plot(rec, prec, label=model_name)
#     ax[2, 0].legend()
#     ax[2, 0].set_xlabel("Recall")
#     ax[2, 0].set_ylabel("Precision")
#     if is_average_folds:
#         ax[2, 0].set_title("Average Precision-Recall curve")
#     elif is_average_bootstraps:
#         ax[2, 0].set_title("Precision-Recall curve with bootstrap uncertainty")
#     else:
#         ax[2, 0].set_title("Precision-Recall curve")
#     ax[2, 0].grid(True)

#     # Plot recall@n vs n
#     for model_name, matrix_folds in zip(model_names, matrix_all):
#         if is_average_folds:
#             av_recall, std_recall = give_average_recall_at_n_folds(
#                 matrix_folds,
#                 n_lst_neg,
#                 bool_test_col=neg_col,
#                 score_col=score_col,
#                 perform_sort=perform_sort,
#                 out_of_matrix_mode=out_of_matrix_mode,
#             )
#             ax[2, 1].plot(n_lst_neg, av_recall, label=model_name)
#             if plot_error_bars:
#                 ax[2, 1].fill_between(n_lst_neg, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#         elif is_average_bootstraps:
#             av_recall, std_recall = give_average_recall_at_n_bootstraps(
#                 matrix_folds,
#                 n_lst_neg,
#                 N_bootstraps,
#                 bool_test_col=neg_col,
#                 score_col=score_col,
#                 perform_sort=perform_sort,
#                 out_of_matrix_mode=out_of_matrix_mode,
#             )
#             ax[2, 1].plot(n_lst_neg, av_recall, label=model_name)
#             if plot_error_bars:
#                 ax[2, 1].fill_between(n_lst_neg, av_recall - std_recall, av_recall + std_recall, alpha=0.2)
#         else:
#             ax[2, 1].plot(
#                 n_lst_neg,
#                 give_recall_at_n(
#                     matrix_folds,
#                     n_lst_neg,
#                     bool_test_col=neg_col,
#                     score_col=score_col,
#                     perform_sort=perform_sort,
#                     out_of_matrix_mode=out_of_matrix_mode,
#                 ),
#             )
#     ax[2, 1].plot([0, n_max_neg], [0, n_max_neg / matrix_length], "k--", label="Random classifier", alpha=0.5)
#     ax[2, 1].set_xlabel("n")
#     if is_average_folds:
#         ax[2, 1].set_ylabel("Average recall@n (negatives)")
#     elif is_average_bootstraps:
#         ax[2, 1].set_ylabel("Average recall@n (negatives) (with bootstrap uncertainty)")
#     else:
#         ax[2, 1].set_ylabel("Recall@n (negatives)")
#     if is_average_folds:
#         ax[2, 1].set_title("Average full matrix Recall@n vs n (negatives - lower is better)")
#     elif is_average_bootstraps:
#         ax[2, 1].set_title("Recall@n (negatives - lower is better) with bootstrap uncertainty")
#     else:
#         ax[2, 1].set_title("Recall@n vs n (negatives - lower is better)")
#     if force_full_y_axis_neg:
#         ax[2, 1].set_ylim(0, 1)
#     ax[2, 1].legend()
#     ax[2, 1].grid(True)

#     # Plot hit@k vs k
#     for model_name, matrix_folds in zip(model_names, matrix_all):
#         if is_average_folds:
#             hit_at_k_dict = give_average_hit_at_k_folds(
#                 matrix_folds, k_max_neg, bool_test_col=neg_col, score_col=score_col
#             )
#             ax[2, 2].plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#             if plot_error_bars:
#                 ax[2, 2].fill_between(
#                     hit_at_k_dict["k"],
#                     hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                     hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                     alpha=0.2,
#                 )
#         elif is_average_bootstraps:
#             hit_at_k_dict = give_average_hit_at_k_bootstraps(
#                 matrix_folds, k_max_neg, N_bootstraps, bool_test_col=neg_col, score_col=score_col
#             )
#             ax[2, 2].plot(hit_at_k_dict["k"], hit_at_k_dict["hit_at_k_mean"], label=model_name)
#             if plot_error_bars:
#                 ax[2, 2].fill_between(
#                     hit_at_k_dict["k"],
#                     hit_at_k_dict["hit_at_k_mean"] - hit_at_k_dict["hit_at_k_std"],
#                     hit_at_k_dict["hit_at_k_mean"] + hit_at_k_dict["hit_at_k_std"],
#                     alpha=0.2,
#                 )
#         else:
#             hit_at_k = give_hit_at_k(matrix_folds, k_max_neg, bool_test_col=neg_col, score_col=score_col)
#             ax[2, 2].plot(hit_at_k["k"], hit_at_k["hit_at_k"], label=model_name)
#     ax[2, 2].plot([0, k_max_neg], [0, k_max_neg / number_of_drugs], "k--", label="Random classifier", alpha=0.5)
#     ax[2, 2].set_xlabel("k")
#     if is_average_folds:
#         ax[2, 2].set_ylabel("Average hit@k (negatives)")
#     elif is_average_bootstraps:
#         ax[2, 2].set_ylabel("Average hit@k (negatives) (with bootstrap uncertainty)")
#     else:
#         ax[2, 2].set_ylabel("Hit@k (negatives)")
#     if is_average_folds:
#         ax[2, 2].set_title("Average disease-specific Hit@k vs k (negatives - lower is better)")
#     elif is_average_bootstraps:
#         ax[2, 2].set_title("Hit@k (negatives - lower is better) with bootstrap uncertainty")
#     else:
#         ax[2, 2].set_title("Disease-specific Hit@k vs k (negatives - lower is better)")
#     if force_full_y_axis_neg:
#         ax[2, 2].set_ylim(0, 1)
#     ax[2, 2].legend()
#     ax[2, 2].grid(True)

#     # Plot entropy@n and commonality between models

#     # Plot drug-entropy@n  and disease-entropy@n vs n
#     for model_name, matrix_folds in zip(model_names, matrix_all):
#         if is_average_folds:
#             entropy_dict = give_average_entropy_at_n(
#                 matrix_folds, n_lst_pos, score_col=score_col, perform_sort=perform_sort
#             )
#             ax[3, 0].plot(n_lst_pos, entropy_dict["drug_entropy_mean"], label=model_name)
#             if plot_error_bars:
#                 ax[3, 0].fill_between(
#                     n_lst_pos,
#                     entropy_dict["drug_entropy_mean"] - entropy_dict["drug_entropy_std"],
#                     entropy_dict["drug_entropy_mean"] + entropy_dict["drug_entropy_std"],
#                     alpha=0.2,
#                 )
#             ax[3, 1].plot(n_lst_pos, entropy_dict["disease_entropy_mean"], label=model_name)
#             if plot_error_bars:
#                 ax[3, 1].fill_between(
#                     n_lst_pos,
#                     entropy_dict["disease_entropy_mean"] - entropy_dict["disease_entropy_std"],
#                     entropy_dict["disease_entropy_mean"] + entropy_dict["disease_entropy_std"],
#                     alpha=0.2,
#                 )
#         else:
#             drug_entropy, disease_entropy = give_entropy_at_n(
#                 matrix_folds, n_lst_pos, score_col=score_col, perform_sort=perform_sort
#             )
#             ax[3, 0].plot(n_lst_pos, drug_entropy, label=model_name)
#             ax[3, 1].plot(n_lst_pos, disease_entropy, label=model_name)

#     # Axis 1 configuration
#     ax[3, 0].legend()
#     ax[3, 0].set_xlabel("n")
#     if is_average_folds:
#         ax[3, 0].set_ylabel("Average drug-entropy@n")
#     else:
#         ax[3, 0].set_ylabel("Drug-entropy@n")
#     ax[3, 0].set_xlim(0, n_max_pos)
#     ax[3, 0].set_ylim(0, 1)
#     if is_average_folds:
#         ax[3, 0].set_title("Average drug-entropy@n vs n")
#     else:
#         ax[3, 0].set_title("Drug-entropy@n vs n")
#     ax[3, 0].grid(True)

#     # Axis 2 configuration
#     ax[3, 1].legend()
#     ax[3, 1].set_xlabel("n")
#     if is_average_folds:
#         ax[3, 1].set_ylabel("Average disease-entropy@n")
#     else:
#         ax[3, 1].set_ylabel("Disease-entropy@n")
#     ax[3, 1].set_xlim(0, n_max_pos)
#     ax[3, 1].set_ylim(0, 1)
#     if is_average_folds:
#         ax[3, 1].set_title("Average disease-entropy@n vs n")
#     else:
#         ax[3, 1].set_title("Disease-entropy@n vs n")
#     ax[3, 1].grid(True)

#     # Plot commonality between models
#     k_lst_commonality = [int(k) for k in np.linspace(1, k_max_commonality + 1, k_steps_commonality)]

#     for (model_name_1, matrix_folds_1), (model_name_2, matrix_folds_2) in combinations(zip(model_names, matrix_all), 2):
#         if is_average_folds:
#             commonality_df = give_average_commonality(
#                 matrix_folds_1, matrix_folds_2, k_lst_commonality, score_col=score_col, perform_sort=perform_sort
#             )
#             ax[3, 2].plot(
#                 k_lst_commonality, commonality_df["commonality_mean"], label=f"{model_name_1} vs {model_name_2}"
#             )
#             if plot_error_bars:
#                 ax[3, 2].fill_between(
#                     k_lst_commonality,
#                     commonality_df["commonality_mean"] - commonality_df["commonality_std"],
#                     commonality_df["commonality_mean"] + commonality_df["commonality_std"],
#                     alpha=0.2,
#                 )
#         else:
#             commonality_df = give_commonality(
#                 matrix_folds_1, matrix_folds_2, k_lst_commonality, score_col=score_col, perform_sort=perform_sort
#             )
#             ax[3, 2].plot(
#                 commonality_df["k"], commonality_df["commonality@k"], label=f"{model_name_1} vs {model_name_2}"
#             )

#     ax[3, 2].legend()
#     ax[3, 2].set_xlabel("k")
#     ax[3, 2].set_ylabel("Commonality@k")
#     ax[3, 2].grid(True)
#     ax[3, 2].set_title("Commonality between models")

#     if sup_title is not None:
#         plt.suptitle(sup_title, y=1.02, fontsize=16)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_distributions_across_folds(
#     matrices_all_folds: Tuple[List[pl.DataFrame]],
#     model_names: Tuple[str],
#     score_col: str = "treat score",
#     pos_col: Optional[str] = "is_known_positive",
#     neg_col: Optional[str] = "is_known_negative",
#     off_col: Optional[str] = "is_off_label",
#     trial_col: Optional[str] = "trial_sig_better",
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots the treat scores across folds for a list of models.

#     Args:
#         matrices_all_folds: A tuple containing lists of matrix data-frames across folds, one for each model.
#         model_names: A list of model names
#         score_col: The column name for the treat score
#         pos_col: The column name for the positive label
#             If None, no positive mean will be plotted
#         neg_col: The column name for the negative label
#             If None, no negative mean will be plotted
#         off_col: The column name for the off-label label
#             If None, no off-label mean will be plotted
#         trial_col: The column name for the trial label
#             If None, no trial mean will be plotted
#         save_path: Path to save the plot to.
#     """
#     n_models = len(matrices_all_folds)
#     n_folds = len(matrices_all_folds[0])
#     _, ax = plt.subplots(n_folds, n_models, figsize=(n_models * 5, n_folds * 5))

#     for i, (model_name, matrix_folds) in enumerate(zip(model_names, matrices_all_folds)):
#         for fold_idx, matrix_fold in enumerate(matrix_folds):
#             ax[fold_idx, i].hist(matrix_fold[score_col], bins=50, alpha=0.7)
#             if pos_col is not None:
#                 ax[fold_idx, i].axvline(
#                     matrix_fold.filter(pl.col(pos_col))[score_col].mean(),
#                     color="red",
#                     linestyle="--",
#                     label="Positive mean",
#                 )
#             if neg_col is not None:
#                 ax[fold_idx, i].axvline(
#                     matrix_fold.filter(pl.col(neg_col))[score_col].mean(),
#                     color="blue",
#                     linestyle="--",
#                     label="Negative mean",
#                 )
#             if off_col is not None:
#                 ax[fold_idx, i].axvline(
#                     matrix_fold.filter(pl.col(off_col))[score_col].mean(),
#                     color="green",
#                     linestyle="--",
#                     label="Off-label mean",
#                 )
#             if trial_col is not None:
#                 ax[fold_idx, i].axvline(
#                     matrix_fold.filter(pl.col(trial_col))[score_col].mean(),
#                     color="orange",
#                     linestyle="--",
#                     label="Trial mean",
#                 )
#             ax[fold_idx, i].set_title(f"{model_name} - Fold {fold_idx}")
#             ax[fold_idx, i].set_xlabel("Treat Score")
#             ax[fold_idx, i].set_ylabel("Count")
#             ax[fold_idx, i].grid(True)
#             ax[fold_idx, i].set_yscale("log")
#             if i == n_models - 1 and fold_idx == 0:
#                 ax[fold_idx, i].legend(bbox_to_anchor=(1.05, 1), loc="upper right")
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_ranking_metrics_over_folds(
#     matrices_all_folds: Tuple[List[pl.DataFrame]],
#     model_names: Tuple[str],
#     bool_test_col: str = "is_known_positive",
#     score_col: str = "treat score",
#     perform_sort: bool = True,
#     n_min: int = 10,
#     n_max: int = 100000,
#     n_steps: int = 1000,
#     k_max: int = 100,
#     sup_title: Optional[str] = None,
#     force_full_y_axis: bool = True,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots recall@n and hit@k across folds for a list of given model.

#     NOTE: This function expects the training set to have been taken out of the matrix dataframes.

#     NOTE: The function will work regardless of whether the matrix dataframes
#     are the same length across folds and models. For the results to be meaningful,
#     the dataframes should represent matrices for consistent lists of drugs and diseases.
#     Some pairs may be missing from the matrices, this is not a problem. Neither is having
#     some additional pairs added for "out-of-matrix" evaluation.

#     Args:
#         matrices_all_folds: A tuple containing lists of matrix data-frames across folds, one for each model.
#             Training set should have been taken out of the matrices.
#             In the case of is_average = False, this is just a tuple of dataframes.
#         model_names: Tuple of model names
#         bool_test_col: Boolean column in the matrix indicating the known positive test set
#         score_col: Column in the matrix containing the treat scores.
#         perform_sort: Whether to sort the matrix by the treat score,
#             or expect the dataframe to be sorted already.
#         n_min: Minimum n value for the recall@n plot
#         n_max: Maximum n value for the recall@n plot
#         n_steps: Number of steps in the recall@n plot
#         k_max: Maximum k value for the hit@k plot
#         sup_title: Title for the plot
#         force_full_y_axis: Whether to force the y-axis to be from 0 to 1 for recall@n and hit@k plots
#         save_path: Path to save the plot to.
#     """
#     # List of n values for recall@n plot
#     n_lst = np.linspace(n_min, n_max, n_steps)
#     n_lst = [int(n) for n in n_lst]

#     # Get the matrix length and number of drugs (minimum among all input matrices)
#     matrix_length = min([len(matrix) for matrix_folds in matrices_all_folds for matrix in matrix_folds])
#     number_of_drugs = min(
#         [len(matrix["source"].unique()) for matrix_folds in matrices_all_folds for matrix in matrix_folds]
#     )

#     # Set up the figure with 2 subplots vertically stacked
#     _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

#     # Plot recall@n vs n
#     color_dict = {}  # Dictionary to store colors for each model
#     for model_name, matrix_folds in zip(model_names, matrices_all_folds):
#         model_total_recall = []
#         # Get a color from the default color cycle for this model
#         color = plt.gca()._get_lines.get_next_color()
#         color_dict[model_name] = color
#         for _, matrix_fold in enumerate(matrix_folds):
#             recall_lst = give_recall_at_n(
#                 matrix_fold, n_lst, bool_test_col=bool_test_col, score_col=score_col, perform_sort=perform_sort
#             )
#             ax1.plot(n_lst, recall_lst, color=color, alpha=0.3)
#             model_total_recall.append(np.array(recall_lst))
#         model_total_recall = np.array(model_total_recall).mean(axis=0)
#         ax1.plot(n_lst, model_total_recall, label=f"{model_name}", color=color, linestyle="--", linewidth=2)

#     ax1.plot([0, matrix_length], [0, 1], "k--", label="Random classifier", alpha=0.5)
#     ax1.legend()
#     ax1.set_xlabel("n")
#     ax1.set_ylabel("Recall@n")
#     ax1.set_xlim(0, n_max)
#     if force_full_y_axis:
#         ax1.set_ylim(0, 1)
#     ax1.set_title("Recall@n vs n across folds")
#     ax1.grid(True)
#     # Plot hit@k vs k
#     for model_name, matrix_folds in zip(model_names, matrices_all_folds):
#         color = color_dict[model_name]
#         for _, matrix_fold in enumerate(matrix_folds):
#             hit_at_k_df = give_hit_at_k(matrix_fold, k_max, bool_test_col=bool_test_col, score_col=score_col)
#             ax2.plot(hit_at_k_df["k"], hit_at_k_df["hit_at_k"], color=color, alpha=0.3)
#         hit_at_k_dict = give_average_hit_at_k_folds(
#             matrix_folds, k_max, bool_test_col=bool_test_col, score_col=score_col
#         )
#         ax2.plot(
#             hit_at_k_dict["k"],
#             hit_at_k_dict["hit_at_k_mean"],
#             label=f"{model_name}",
#             color=color,
#             linestyle="--",
#             linewidth=2,
#         )

#     ax2.plot([0, number_of_drugs], [0, 1], "k--", label="Random classifier", alpha=0.5)
#     ax2.legend()
#     ax2.set_xlabel("k")
#     ax2.set_ylabel("Hit@k")
#     ax2.set_title("Hit@k vs k across folds")
#     ax2.set_xlim(0, k_max)
#     if force_full_y_axis:
#         ax2.set_ylim(0, 1)
#     ax2.grid(True)

#     plt.suptitle(sup_title)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)


# def plot_commonality_between_folds(
#     matrices_all_folds: Tuple[List[pl.DataFrame]],
#     model_names: Tuple[str],
#     score_col: str = "treat score",
#     perform_sort: bool = True,
#     k_max: int = 100000,
#     k_steps: int = 1000,
#     sup_title: Optional[str] = "Commonality between folds",
#     plot_error_bars: bool = True,
#     save_path: Optional[str] = None,
# ) -> None:
#     """Plots average commonality@k across folds vs k curves for a list of models.

#     Args:
#         matrices_all_folds: A tuple containing lists of matrix data-frames across folds, one for each model.
#         model_names: A list of model names
#         score_col: The column name for the treat score
#         perform_sort: Whether to sort the matrix by the treat score,
#             or expect the dataframe to be sorted already.
#         k_max: Maximum k value for the commonality@k plot
#         k_steps: Number of steps in the commonality@k plot
#         sup_title: Title for the plot
#         plot_error_bars: Whether to plot the error bars
#         save_path: Path to save the plot to.
#     """
#     k_lst = [int(k) for k in np.linspace(1, k_max + 1, k_steps)]

#     # Sort matrices if requested
#     if perform_sort:
#         for matrix_folds in matrices_all_folds:
#             for matrix in matrix_folds:
#                 matrix = matrix.sort(by=score_col, descending=True)

#     _, ax1 = plt.subplots(1, 1, figsize=(10, 5))

#     # Calculate and plot commonality between folds
#     for model_name, matrix_folds in zip(model_names, matrices_all_folds):
#         commonality_list = []
#         for fold_1, fold_2 in combinations(matrix_folds, 2):
#             commonality_df = give_commonality(fold_1, fold_2, k_lst, perform_sort=False)
#             commonality_list.append(commonality_df["commonality@k"].to_list())
#         commonality_mean = np.mean(commonality_list, axis=0)
#         commonality_std = np.std(commonality_list, axis=0)
#         ax1.plot(k_lst, commonality_mean, label=f"{model_name}")
#         if plot_error_bars:
#             ax1.fill_between(k_lst, commonality_mean - commonality_std, commonality_mean + commonality_std, alpha=0.2)

#     ax1.legend()
#     ax1.set_xlabel("k")
#     ax1.set_ylabel("Commonality@k")
#     ax1.grid(True)
#     ax1.set_ylim(0, 1)
#     plt.suptitle(sup_title)
#     plt.tight_layout()
#     plt.show()

#     if save_path is not None:
#         plt.savefig(save_path)
