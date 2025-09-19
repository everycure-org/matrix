"""Module containing polars-based implementations for evaluation metrics and plotting utilities.

Intended to be used as a library for notebook-based analysis.
"""

from typing import List, Tuple

import numpy as np
import polars as pl
from scipy.stats import entropy
from sklearn.metrics import auc, f1_score, precision_recall_curve
from tqdm import tqdm

# ---
# Full matrix ranking
# ---


def give_recall_at_n(
    matrix: pl.DataFrame,
    n_lst: list[int],
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    perform_sort: bool = True,
    out_of_matrix_mode: bool = False,
) -> List[float]:
    """
    Returns the recall@n score for a list of n values.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        n_lst: List of n values to calculate the recall@n score for.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
            In this case, the matrix dataframe must also contain a boolean column "in_matrix".
    Returns:
        A list of recall@n scores for the list of n values.
    """
    # Number of known positives
    N = len(matrix.filter(pl.col(bool_test_col)))
    if N == 0:
        return [0] * len(n_lst)

    if out_of_matrix_mode:
        matrix = matrix.filter(pl.col("in_matrix") | pl.col(bool_test_col))

    # Sort by treat score
    if perform_sort or out_of_matrix_mode:
        matrix = matrix.sort(by=score_col, descending=True)

    # Ranks of the known positives
    ranks_series = matrix.with_row_index("index").filter(pl.col(bool_test_col)).select(pl.col("index")).to_series() + 1

    # Recall@n scores
    recall_lst = [(ranks_series <= n).sum() / N for n in n_lst]

    return recall_lst


def give_recall_at_n_folds(
    matrix_folds: List[pl.DataFrame],
    n_lst: List[int],
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    perform_sort: bool = True,
    out_of_matrix_mode: bool = False,
) -> List[np.ndarray]:
    """
    Returns the recall@n scores for a list of n values over a list of folds.

    Args:
        matrix_folds: List of drug-disease treat score matrix dataframes for each fold
            Training set should have been taken out of the matrices.
        n_lst: List of n values to calculate the recall@n score for.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
            In this case, the matrix dataframes must also contain boolean columns "in_matrix".

    Returns:
        A 2d array of recall@n scores for the list of n values over the list of folds.
        The first dimension is the fold, the second is the n value.
    """
    return np.array(
        [
            give_recall_at_n(
                fold,
                n_lst,
                bool_test_col=bool_test_col,
                score_col=score_col,
                perform_sort=perform_sort,
                out_of_matrix_mode=out_of_matrix_mode,
            )
            for fold in matrix_folds
        ]
    )


def give_average_recall_at_n_folds(
    matrix_folds: List[pl.DataFrame],
    n_lst: List[int],
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    perform_sort: bool = True,
    give_std: bool = True,
    out_of_matrix_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the average and standard deviation of the recall@n scores for a list of n values over a list of folds.

    FUTURE: Make output a dict, consistent with entropy@n and hit@k functions.

    Args:
        matrix_folds: List of drug-disease treat score matrix dataframes for each fold
            Training set should have been taken out of the matrices.
        n_lst: List of n values to calculate the average recall@n score for.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        give_std: Whether to return the standard deviation of the recall@n scores.
        out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
            In this case, the matrix dataframes must also contain boolean columns "in_matrix".
    Returns:
        A tuple of two arrays, the first containing the average recall@n scores and the second containing the standard deviation of the recall@n scores.
    """
    all_recalls_arr = give_recall_at_n_folds(
        matrix_folds,
        n_lst,
        bool_test_col=bool_test_col,
        score_col=score_col,
        perform_sort=perform_sort,
        out_of_matrix_mode=out_of_matrix_mode,
    )
    return all_recalls_arr.mean(axis=0), (all_recalls_arr.std(axis=0) if give_std else None)


def give_recall_at_n_bootstraps(
    matrix: pl.DataFrame,
    n_lst: list[int],
    N_bootstraps: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    perform_sort: bool = True,
    out_of_matrix_mode: bool = False,
    seed: int = 42,
):
    """
    Returns recall@n score for a list of n values, with bootstrap uncertainty estimation.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        n_lst: List of n values to calculate the recall@n score for.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
            In this case, the matrix dataframe must also contain a boolean column "in_matrix".
        seed: Seed for the bootstrap resampling.
    Returns:
        A 2d array of recall@n scores for the list of n values.
        The first dimension is the bootstrap, the second is the n value.
    """
    pl.set_random_seed(seed)

    # Number of known positives
    N = len(matrix.filter(pl.col(bool_test_col)))
    if N == 0:
        return np.zeros((N_bootstraps, len(n_lst)))

    if out_of_matrix_mode:
        matrix = matrix.filter(pl.col("in_matrix") | pl.col(bool_test_col))

    # Sort by treat score
    if perform_sort:
        matrix = matrix.sort(by=score_col, descending=True)

    # Ranks of the known positives
    ranks_series = matrix.with_row_index("index").filter(pl.col(bool_test_col)).select(pl.col("index")).to_series() + 1

    # Function for resampling and calculating recall@n
    def bootstrap_recall_lst(ranks_series, N, n_lst):
        ranks_series_resampled = ranks_series.sample(N, with_replacement=True).sort(descending=False)
        return [(ranks_series_resampled <= n).sum() / N for n in n_lst]

    # Calculate recall@n for each bootstrap
    return np.array([bootstrap_recall_lst(ranks_series, N, n_lst) for _ in range(N_bootstraps)])


def give_average_recall_at_n_bootstraps(
    matrix: pl.DataFrame,
    n_lst: list[int],
    N_bootstraps: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    perform_sort: bool = True,
    give_std: bool = True,
    out_of_matrix_mode: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the average and standard deviation of the recall@n scores for a list of n values, using bootstrap uncertainty estimation.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        n_lst: List of n values to calculate the average recall@n score for.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score, or expect the dataframe to be sorted already.
        give_std: Whether to return the standard deviation of the recall@n scores.
        out_of_matrix_mode: Whether to use the out of matrix mode, where pairs outside the matrix may be used in the calculation.
            In this case, the matrix dataframe must also contain a boolean column "in_matrix".
        seed: Seed for the bootstrap resampling.
    Returns:
        A tuple of two arrays, the first containing the average recall@n scores and the second containing the standard deviation of the recall@n scores.
    """
    all_recalls_arr = give_recall_at_n_bootstraps(
        matrix,
        n_lst,
        N_bootstraps,
        bool_test_col=bool_test_col,
        score_col=score_col,
        perform_sort=perform_sort,
        out_of_matrix_mode=out_of_matrix_mode,
        seed=seed,
    )

    return all_recalls_arr.mean(axis=0), (all_recalls_arr.std(axis=0) if give_std else None)


# ---
# Disease-specific ranking
# ---


def give_hit_at_k(
    matrix: pl.DataFrame,
    k_max: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
) -> pl.DataFrame:
    """
    Returns the hit@k score for a list of k values.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        k_max: Maximum k value to compute hit@k for
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
    Returns:
        A dataframe with the hit@k scores and the k values.
    """
    # Restrict to test diseases
    test_diseases = (
        matrix.group_by("target")
        .agg(pl.col(bool_test_col).sum().alias("num_known_positives"))
        .filter(pl.col("num_known_positives") > 0)
        .select(pl.col("target"))
        .to_series()
        .to_list()
    )
    matrix = matrix.filter(pl.col("target").is_in(test_diseases))

    # Add disease-specific ranks
    matrix = matrix.with_columns(disease_rank=pl.col(score_col).rank(descending=True, method="random").over("target"))

    #  Remove other positives from ranking
    matrix = matrix.filter(pl.col(bool_test_col)).with_columns(
        disease_rank_among_positives=pl.col(score_col).rank(descending=True, method="dense").over("target")
    )
    matrix = matrix.with_columns(
        disease_rank_against_negatives=pl.col("disease_rank") - pl.col("disease_rank_among_positives") + 1
    )

    # Count number of positives at each rank and cumulative sum
    ranks_for_test_set = (
        matrix.filter(pl.col(bool_test_col))
        .group_by("disease_rank_against_negatives")
        .len()
        .sort("disease_rank_against_negatives")
    )
    ranks_for_test_set = ranks_for_test_set.with_columns(pl.col("len").cum_sum().alias("cumulative_len"))

    # Compute hit@k for each k
    df_hit_at_k = pl.DataFrame(
        {
            "k": ranks_for_test_set["disease_rank_against_negatives"],
            "hit_at_k": ranks_for_test_set["cumulative_len"] / len(matrix.filter(pl.col(bool_test_col))),
        }
    )

    return df_hit_at_k.filter(pl.col("k") <= k_max)


def give_hit_at_k_folds(
    matrix_folds: List[pl.DataFrame],
    k_max: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the hit@k score for a list of k values over a list of folds.

    Args:
        matrix_folds: List of drug-disease treat score matrix dataframes for each fold
            Training set should have been taken out of the matrices.
        k_max: Maximum k value to compute hit@k for
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.

    Returns:
        Tuple of two arrays:
            1. A 2d numpy array of the hit@k scores for the list of k values over the folds.
                The first dimension is the fold, the second is the k value.
            2. A numpy array of the k values.
    """
    # Compute Hit@k
    hit_at_k_folds = [
        give_hit_at_k(fold, k_max, bool_test_col=bool_test_col, score_col=score_col) for fold in matrix_folds
    ]

    # Prepend value 0 for k=0
    hit_at_k_folds = [
        pl.concat([pl.DataFrame({"k": [0], "hit_at_k": [0]}).cast({"k": pl.UInt32, "hit_at_k": pl.Float64}), fold])
        for fold in hit_at_k_folds
    ]
    # Join to fill missing k values
    hit_at_k_folds = [
        fold.join(pl.DataFrame({"k": list(range(0, k_max + 1))}).cast(pl.UInt32), on="k", how="right").fill_null(
            strategy="forward"
        )
        for fold in hit_at_k_folds
    ]

    k_lst = hit_at_k_folds[0]["k"].to_numpy()

    return np.array([fold["hit_at_k"].to_numpy() for fold in hit_at_k_folds]), k_lst


def give_average_hit_at_k_folds(
    matrix_folds: List[pl.DataFrame],
    k_max: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
) -> dict[str, np.ndarray]:
    """
    Returns the average and std (over folds) of the hit@k score for a list of k values.

    Args:
        matrix_folds: List of drug-disease treat score matrix dataframes for each fold
            Training set should have been taken out of the matrices.
        k_max: Maximum k value to compute hit@k for
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
    Returns:
        A dictionary with the average and std (over folds) of the hit@k scores and the k values.
    """
    all_hit_at_k_arr, k_lst = give_hit_at_k_folds(matrix_folds, k_max, bool_test_col=bool_test_col, score_col=score_col)
    # Compute average Hit@k
    return {"hit_at_k_mean": all_hit_at_k_arr.mean(axis=0), "hit_at_k_std": all_hit_at_k_arr.std(axis=0), "k": k_lst}


def give_hit_at_k_bootstraps(
    matrix_df: pl.DataFrame,
    k_max: int,
    N_bootstraps: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    seed: int = 42,
) -> np.ndarray:
    """
    Returns the hit@k score for a list of k values, using bootstrap uncertainty estimation.

    Args:
        matrix_df: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        k_max: Maximum k value to compute hit@k for
        N_bootstraps: Number of bootstraps to compute
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        seed: Seed for the bootstrap resampling.

    Returns:
        A 2d numpy array of the hit@k scores for the list of k values over the bootstraps.
        The first dimension is the bootstrap, the second is the k value.
    """
    pl.set_random_seed(seed)

    # Restrict to test diseases
    test_diseases = (
        matrix_df.group_by("target")
        .agg(pl.col(bool_test_col).sum().alias("num_known_positives"))
        .filter(pl.col("num_known_positives") > 0)
        .select(pl.col("target"))
        .to_series()
        .to_list()
    )
    matrix_df = matrix_df.filter(pl.col("target").is_in(test_diseases))

    # Add disease-specific ranks
    matrix_df = matrix_df.with_columns(
        disease_rank=pl.col(score_col).rank(descending=True, method="random").over("target")
    )

    #  Remove other positives from ranking
    matrix_df = matrix_df.filter(pl.col(bool_test_col)).with_columns(
        disease_rank_among_positives=pl.col(score_col).rank(descending=True, method="dense").over("target")
    )
    matrix_df = matrix_df.with_columns(
        disease_rank_against_negatives=pl.col("disease_rank") - pl.col("disease_rank_among_positives") + 1
    )

    # Restrict to ground truth pairs
    ground_truth_pairs = matrix_df.filter(pl.col(bool_test_col))
    N_ground_truth = len(ground_truth_pairs)

    # Compute hit@k list for each bootstrap
    bootstraps = []
    for _ in range(N_bootstraps):
        ground_truth_pairs_resampled = ground_truth_pairs.sample(N_ground_truth, with_replacement=True)
        ranks_for_test_set = (
            ground_truth_pairs_resampled.group_by("disease_rank_against_negatives")
            .len()
            .sort("disease_rank_against_negatives")
        )
        ranks_for_test_set = ranks_for_test_set.with_columns(pl.col("len").cum_sum().alias("cumulative_len"))
        df_hit_at_k = pl.DataFrame(
            {
                "k": ranks_for_test_set["disease_rank_against_negatives"],
                "hit_at_k": ranks_for_test_set["cumulative_len"] / len(matrix_df.filter(pl.col(bool_test_col))),
            }
        )

        # Prepend value 0 for k=0
        df_hit_at_k = pl.concat(
            [pl.DataFrame({"k": [0], "hit_at_k": [0]}).cast({"k": pl.UInt32, "hit_at_k": pl.Float64}), df_hit_at_k]
        )

        # Join to fill missing k values
        df_hit_at_k = df_hit_at_k.join(
            pl.DataFrame({"k": list(range(0, k_max + 1))}).cast(pl.UInt32), on="k", how="right"
        ).fill_null(strategy="forward")

        bootstraps.append(df_hit_at_k["hit_at_k"].to_numpy())

    return np.array(bootstraps)


def give_average_hit_at_k_bootstraps(
    matrix_df: pl.DataFrame,
    k_max: int,
    N_bootstraps: int,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Returns the average and std (over bootstraps) of the hit@k score for a list of k values.

    Args:
        matrix_df: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        k_max: Maximum k value to compute hit@k for
        N_bootstraps: Number of bootstraps to compute
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
        seed: Seed for the bootstrap resampling.

    Returns:
        A dictionary with the average and std (over bootstraps) of the hit@k scores and the k values.
    """
    all_hit_at_k_arr = give_hit_at_k_bootstraps(
        matrix_df, k_max, N_bootstraps, bool_test_col=bool_test_col, score_col=score_col, seed=seed
    )
    return {
        "hit_at_k_mean": all_hit_at_k_arr.mean(axis=0),
        "hit_at_k_std": all_hit_at_k_arr.std(axis=0),
        "k": list(range(0, k_max + 1)),
    }


def give_disease_specific_mrr(
    matrix: pl.DataFrame,
    bool_test_col: str = "is_known_positive",
    score_col: str = "treat score",
) -> pl.DataFrame:
    """
    Returns the MRR score for disease-specific ranking.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        bool_test_col: Boolean column in the matrix indicating the known positive test set
        score_col: Column in the matrix containing the treat scores.
    Returns:
        The MRR score for disease-specific ranking.
    """
    # Restrict to test diseases
    test_diseases = (
        matrix.group_by("target")
        .agg(pl.col(bool_test_col).sum().alias("num_known_positives"))
        .filter(pl.col("num_known_positives") > 0)
        .select(pl.col("target"))
        .to_series()
        .to_list()
    )
    matrix = matrix.filter(pl.col("target").is_in(test_diseases))

    # Add disease-specific ranks
    matrix = matrix.with_columns(disease_rank=pl.col(score_col).rank(descending=True, method="random").over("target"))

    #  Remove other positives from ranking
    matrix = matrix.filter(pl.col(bool_test_col)).with_columns(
        disease_rank_among_positives=pl.col(score_col).rank(descending=True, method="dense").over("target")
    )
    matrix = matrix.with_columns(
        disease_rank_against_negatives=pl.col("disease_rank") - pl.col("disease_rank_among_positives") + 1
    )

    return (1 / matrix["disease_rank_against_negatives"]).mean()


# ---
# Classification metrics
# ---


def give_f1_score(
    matrix: pl.DataFrame,
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
    threshold: float = 0.5,
) -> float:
    """
    Returns the F1 score for the matrix.
    """
    ground_truth = matrix.filter(pl.col(bool_test_col_pos) | pl.col(bool_test_col_neg)).select(
        bool_test_col_pos, score_col
    )
    return f1_score(ground_truth[bool_test_col_pos], ground_truth[score_col] > threshold)


def give_precision_recall_curve(
    matrix: pl.DataFrame,
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
) -> pl.DataFrame:
    """
    Returns the precision-recall curve for a matrix dataframe of drug-disease predictions.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        bool_test_col_pos: Boolean column in the matrix indicating the known positive test set
        bool_test_col_neg: Boolean column in the matrix indicating the known negative test set
        score_col: Column in the matrix containing the treat scores.

    Returns:
        A tuple of two arrays, the first containing the precision and the second containing the recall.
    """
    ground_truth = matrix.filter(pl.col(bool_test_col_pos) | pl.col(bool_test_col_neg)).select(
        bool_test_col_pos, score_col
    )
    precision, recall, _ = precision_recall_curve(ground_truth[bool_test_col_pos], ground_truth[score_col])
    return precision, recall


def give_average_precision_recall_curve_folds(
    matrices: List[pl.DataFrame],
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
    num_points: int = 1000,
) -> pl.DataFrame:
    """
    Returns the height of the average and std (over folds) of the precision-recall curve for a
    list of input dataframes with treat scores for each fold.

    Args:
        matrices: List of dataframes with treat scores for each fold.
            Training set should have been taken out of the matrices.
        bool_test_col_pos: Boolean column in the matrix indicating the known positive test set
        bool_test_col_neg: Boolean column in the matrix indicating the known negative test set
        score_col: Column in the matrix containing the treat scores.

    Returns:
        A tuple of three arrays,
            1. the first containing the x values,
            2. the second containing the height of the average precision-recall curve,
            3. the third containing the std of the precision-recall curve.
    """
    x_vals = np.linspace(0, 1, num_points)
    precision_recall_curve_heights_lst = []
    for fold in matrices:
        # Get precision and recall for fold
        precision, recall = give_precision_recall_curve(fold, bool_test_col_pos, bool_test_col_neg, score_col)

        # Interpolate to get height of the fold precision-recall curve at each x value
        sort_idx = np.argsort(recall)
        recall = recall[sort_idx]
        precision = precision[sort_idx]
        precision_recall_curve_heights_lst.append(np.interp(x_vals, recall, precision))

    # Compute mean height
    precision_recall_curve_heights_av = np.array(precision_recall_curve_heights_lst).mean(axis=0)
    precision_recall_curve_heights_std = np.array(precision_recall_curve_heights_lst).std(axis=0)

    return x_vals, precision_recall_curve_heights_av, precision_recall_curve_heights_std


def give_average_precision_recall_curve_bootstraps(
    matrix_df: pl.DataFrame,
    N_bootstraps: int = 1000,
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
    num_points: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Returns the height of the average and std (using bootstrap estimation) of the precision-recall curve for a
    list of input dataframes with treat scores for each fold.

    Args:
        matrix_df: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrix.
        N_bootstraps: Number of bootstraps to compute
        bool_test_col_pos: Boolean column in the matrix indicating the known positive test set
        bool_test_col_neg: Boolean column in the matrix indicating the known negative test set
        score_col: Column in the matrix containing the treat scores.
        num_points: Number of points to sample from the precision-recall curve.
        seed: Seed for the bootstrap resampling.

    Returns:
        A tuple of three arrays,
            1. the first containing the x values,
            2. the second containing the height of the average precision-recall curve,
            3. the third containing the std of the precision-recall curve.
    """
    pl.set_random_seed(seed)

    x_vals = np.linspace(0, 1, num_points)
    ground_truth = matrix_df.filter(pl.col(bool_test_col_pos) | pl.col(bool_test_col_neg)).select(
        bool_test_col_pos, score_col
    )
    N_ground_truth = len(ground_truth)
    precision_recall_curve_heights_lst = []
    for _ in range(N_bootstraps):
        # Sample and compute precision-recall curve
        ground_truth_sample = ground_truth.sample(N_ground_truth, with_replacement=True)
        precision, recall, _ = precision_recall_curve(
            ground_truth_sample[bool_test_col_pos], ground_truth_sample[score_col]
        )

        # Interpolate to get height of the fold precision-recall curve at each x value
        sort_idx = np.argsort(recall)
        recall = recall[sort_idx]
        precision = precision[sort_idx]
        precision_recall_curve_heights_lst.append(np.interp(x_vals, recall, precision))

    # Compute mean height
    precision_recall_curve_heights_av = np.array(precision_recall_curve_heights_lst).mean(axis=0)
    precision_recall_curve_heights_std = np.array(precision_recall_curve_heights_lst).std(axis=0)

    return x_vals, precision_recall_curve_heights_av, precision_recall_curve_heights_std


def give_auprc(
    matrix: pl.DataFrame,
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
) -> float:
    """
    Returns AUPRC for a matrix dataframe of drug-disease predictions.

    Args:
        matrix: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrices.
        bool_test_col_pos: Boolean column in the matrix indicating the known positive test set
        bool_test_col_neg: Boolean column in the matrix indicating the known negative test set
        score_col: Column in the matrix containing the treat scores.
    """
    precision, recall = give_precision_recall_curve(matrix, bool_test_col_pos, bool_test_col_neg, score_col)
    return auc(recall, precision)


def give_average_auprc_folds(
    matrices: list[pl.DataFrame],
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
) -> dict:
    """
    Returns average and standard deviation for AUPRC over folds.

    Args:
        matrices: List of dataframes with treat scores for each fold.
            Training set should have been taken out of the matrices.
        bool_test_col_pos: Boolean column in the matrix indicating the known positive test set
        bool_test_col_neg: Boolean column in the matrix indicating the known negative test set
        score_col: Column in the matrix containing the treat scores.

    Returns:
        Dictionary with results
    """
    auprc_list = [give_auprc(matrix, bool_test_col_pos, bool_test_col_neg, score_col) for matrix in matrices]
    return {
        "auprc_mean": np.array(auprc_list).mean(),
        "auprc_std": np.array(auprc_list).std(),
    }


def give_average_auprc_bootstraps(
    matrix_df: pl.DataFrame,
    N_bootstraps: int = 1000,
    bool_test_col_pos: str = "is_known_positive",
    bool_test_col_neg: str = "is_known_negative",
    score_col: str = "treat score",
    num_points: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Returns AUPRC with bootstrap estimates.

    NOTE: Accepting code duplication here with give_average_precision_recall_curve_bootstraps.
    Clean-up required upon productionisation.

    Args:
        matrix_df: Dataframe of drug-disease pairs with treat scores.
            Training set should have been taken out of the matrix.
        N_bootstraps: Number of bootstraps to compute
        bool_test_col_pos: Boolean column in the matrix indicating the known positive test set
        bool_test_col_neg: Boolean column in the matrix indicating the known negative test set
        score_col: Column in the matrix containing the treat scores.
        num_points: Number of points to sample from the precision-recall curve.
        seed: Seed for the bootstrap resampling.

    Returns:
        Dictionary with results
    """
    x_vals = np.linspace(0, 1, num_points)
    ground_truth = matrix_df.filter(pl.col(bool_test_col_pos) | pl.col(bool_test_col_neg)).select(
        bool_test_col_pos, score_col
    )
    N_ground_truth = len(ground_truth)
    precision_recall_curve_heights_lst = []
    for _ in range(N_bootstraps):
        # Sample and compute precision-recall curve
        ground_truth_sample = ground_truth.sample(N_ground_truth, with_replacement=True, seed=seed)
        precision, recall, _ = precision_recall_curve(
            ground_truth_sample[bool_test_col_pos], ground_truth_sample[score_col]
        )

        # Interpolate to get height of the fold precision-recall curve at each x value
        sort_idx = np.argsort(recall)
        recall = recall[sort_idx]
        precision = precision[sort_idx]
        precision_recall_curve_heights_lst.append(np.interp(x_vals, recall, precision))

    # Compute AUC for each bootstrap
    auc_lst = [
        auc(x_vals, precision_recall_curve_heights)
        for precision_recall_curve_heights in tqdm(precision_recall_curve_heights_lst)
    ]

    return {
        "auprc_mean": np.array(auc_lst).mean(),
        "auprc_std": np.array(auc_lst).mean(),
    }


# ---
# Stability metrics
# ---


def give_commonality(
    matrix_1: pl.DataFrame,
    matrix_2: pl.DataFrame,
    k_lst: list[int],
    score_col: str = "treat score",
    perform_sort: bool = True,
) -> pl.DataFrame:
    """Calculate the commonality@k for a list of k values.

    Args:
        matrix_1: First matrix to compare.
        matrix_2: Second matrix to compare.
        k_lst: List of k values to calculate commonality@k for.
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score.

    Returns:
        DataFrame with k values and commonality@k.
    """
    if perform_sort:
        matrix_1 = matrix_1.sort(by=score_col, descending=True)
        matrix_2 = matrix_2.sort(by=score_col, descending=True)

    # Restrict to required columns and rows
    n_max = max(k_lst)
    matrix_1 = matrix_1.select(pl.col("source"), pl.col("target")).head(n_max)
    matrix_2 = matrix_2.select(pl.col("source"), pl.col("target")).head(n_max)

    # Add rank columns and joining
    matrix_union = pl.concat([matrix_1, matrix_2]).unique()
    matrix_1 = matrix_1.with_row_index(name="matrix_1_rank", offset=1).join(
        matrix_union, on=["source", "target"], how="right"
    )
    matrix_2 = matrix_2.with_row_index(name="matrix_2_rank", offset=1).join(
        matrix_union, on=["source", "target"], how="right"
    )
    matrix_joined = matrix_1.join(matrix_2, on=["source", "target"], how="inner")

    # Calculate commonality
    commonality_df = (
        pl.DataFrame(
            {
                "k": k_lst,
                "n_common": [
                    matrix_joined.filter((pl.col("matrix_1_rank") < k) & (pl.col("matrix_2_rank") < k)).height
                    for k in k_lst
                ],
            }
        )
        .with_columns((pl.col("n_common") / pl.col("k")).alias("commonality@k").cast(pl.Float64))
        .select(pl.col("k"), pl.col("commonality@k"))
    )
    return commonality_df


def give_average_commonality(
    matrices_1: List[pl.DataFrame],
    matrices_2: List[pl.DataFrame],
    k_lst: list[int],
    score_col: str = "treat score",
    perform_sort: bool = True,
) -> pl.DataFrame:
    """Calculate the average commonality@k for a list of matrices.

    Args:
        matrices_1: List of matrix dataframes across folds for model 1.
        matrices_2: List of matrix dataframes across folds for model 2.
        k_lst: List of k values to calculate commonality@k for.
        score_col: Column in the matrix containing the treat scores.
        perform_sort: Whether to sort the matrix by the treat score.

    Returns:
        Dictionary with average and std of commonality@k for each k.
    """
    commonality_lst = []
    for matrix_1, matrix_2 in zip(matrices_1, matrices_2):
        commonality_df = give_commonality(matrix_1, matrix_2, k_lst, score_col=score_col, perform_sort=perform_sort)
        commonality_lst.append(commonality_df["commonality@k"].to_numpy())
    return {
        "k": k_lst,
        "commonality_mean": np.array(commonality_lst).mean(axis=0),
        "commonality_std": np.array(commonality_lst).std(axis=0),
    }


# ---
# Frequent flyer metrics
# ---


def give_appearance_at_n(
    matrix: pl.DataFrame, n_lst: list[int], perform_sort: bool = True, score_col: str = "treat score"
) -> tuple[list[float], list[float]]:
    """
    Give the drug and disease appearance@n scores for a list of n values.

    Args:
        matrix: Dataframe representing a matrix of drug-disease pairs
        n_lst: The list of n values to calculate the appearance@n score for.
            Must be sorted in ascending order.
        perform_sort: Whether to sort the matrix by the treat score.
        score_col: Column in the matrix containing the treat scores.

    Returns:
        A tuple of two lists, the first containing the drug appearance@n scores and the second containing the disease appearance@n scores.
    """
    # Sort by treat score
    if perform_sort:
        matrix = matrix.sort(by=score_col, descending=True)

    # Total number of unique drugs and diseases
    n_drugs = matrix.select(pl.col("source").n_unique()).to_series().to_list()[0]
    n_diseases = matrix.select(pl.col("target").n_unique()).to_series().to_list()[0]

    # Calculate the number of unique drugs and diseases for each n
    unique_drugs = set()
    unique_diseases = set()
    num_drugs_lst = []
    num_diseases_lst = []

    n_lst = [0] + n_lst
    for i in tqdm(range(len(n_lst) - 1)):
        matrix_slice = matrix.slice(n_lst[i], n_lst[i + 1])
        unique_drugs = unique_drugs.union(set(matrix_slice.select(pl.col("source")).to_series()))
        unique_diseases = unique_diseases.union(set(matrix_slice.select(pl.col("target")).to_series()))
        num_drugs_lst.append(len(unique_drugs))
        num_diseases_lst.append(len(unique_diseases))

    # Calculate appearance@n for each n
    proportion_drugs_lst = [num / n_drugs for num in num_drugs_lst]
    proportion_diseases_lst = [num / n_diseases for num in num_diseases_lst]

    return proportion_drugs_lst, proportion_diseases_lst


def give_entropy_at_n(
    matrix: pl.DataFrame, n_lst: list[int], perform_sort: bool = True, score_col: str = "treat score"
) -> tuple[list[float], list[float]]:
    """
    Give the drug and disease entropy@n scores for a list of n values.

    Args:
        matrix: Dataframe representing a matrix of drug-disease pairs
        n_lst: The list of n values to calculate the entropy@n score for.
            Must be sorted in ascending order.
        perform_sort: Whether to sort the matrix by the treat score.
        score_col: Column in the matrix containing the treat scores.
    Returns:
        A tuple of two lists, the first containing the drug entropy@n scores and the second containing the disease entropy@n scores.
    """
    # Sort by treat score
    if perform_sort:
        matrix = matrix.sort(by=score_col, descending=True)

    # Total number of unique drugs and diseases
    n_drugs = matrix.select(pl.col("source").n_unique()).to_series().to_list()[0]
    n_diseases = matrix.select(pl.col("target").n_unique()).to_series().to_list()[0]
    drug_entropy_lst = []
    disease_entropy_lst = []

    # Initialize count DataFrames with all unique drugs/diseases
    drug_count = matrix.select("source").unique().with_columns(pl.lit(0).alias("count"))
    disease_count = matrix.select("target").unique().with_columns(pl.lit(0).alias("count"))

    n_lst = [0] + n_lst
    # for i in tqdm(range(len(n_lst) - 1)):
    for i in range(len(n_lst) - 1):
        matrix_slice = matrix.slice(n_lst[i], n_lst[i + 1] - n_lst[i])

        def update_count(count_df: pl.DataFrame, matrix_slice: pl.DataFrame, col_name: str) -> pl.DataFrame:
            """
            Update the count given new elements in the matrix slice.
            """
            slice_count = matrix_slice.select(pl.col(col_name)).to_series().value_counts(name="count_new")

            count_df = (
                count_df.join(slice_count, on=col_name, how="left")
                .with_columns(pl.col("count_new").fill_null(0))
                .with_columns((pl.col("count") + pl.col("count_new")).alias("count"))
                .drop("count_new")
            )

            return count_df

        drug_count = update_count(drug_count, matrix_slice, "source")
        disease_count = update_count(disease_count, matrix_slice, "target")

        drug_entropy_lst.append(entropy(drug_count.select("count").to_numpy().flatten(), base=n_drugs))
        disease_entropy_lst.append(entropy(disease_count.select("count").to_numpy().flatten(), base=n_diseases))

    return drug_entropy_lst, disease_entropy_lst


def give_average_entropy_at_n(
    matrices: List[pl.DataFrame], n_lst: list[int], perform_sort: bool = True, score_col: str = "treat score"
) -> tuple[list[float], list[float]]:
    """
    Give the average entropy@n score over folds for a list of n values.

    Args:
        matrices: List of matrix dataframes across folds.
        n_lst: List of n values to calculate the average entropy@n score for.
            Must be sorted in ascending order.
    Returns:
        Dictionary with average and std of drug and disease entropy@n scores for each n.
    """
    drug_entropy_lst_all = []
    disease_entropy_lst_all = []
    for matrix in matrices:
        drug_entropy_lst, disease_entropy_lst = give_entropy_at_n(
            matrix, n_lst, perform_sort=perform_sort, score_col=score_col
        )
        drug_entropy_lst_all.append(drug_entropy_lst)
        disease_entropy_lst_all.append(disease_entropy_lst)
    return {
        "drug_entropy_mean": np.array(drug_entropy_lst_all).mean(axis=0),
        "disease_entropy_mean": np.array(disease_entropy_lst_all).mean(axis=0),
        "drug_entropy_std": np.array(drug_entropy_lst_all).std(axis=0),
        "disease_entropy_std": np.array(disease_entropy_lst_all).std(axis=0),
    }


# ---
# Out of matrix evaluation
# ---


def give_extra_datasets(
    n_folds: int,
    splits: pl.DataFrame,
    drugs: pl.DataFrame,
    extra_datasets: List[pl.DataFrame],
    extra_datasets_names: List[str],
) -> List[pl.DataFrame]:
    """ "Give extra dataset for out of matrix evaluation.

    This is based on constructing a test set from multiple sources. Then the dataset is:
        all drugs x all test diseases + test set

    Assumes that the splits dataframe comes form a single source of data (i.e. on label), which will likely not hold at some point in the future.
    Args:
        splits: On-label dataframe
            Must contain "source", "target", "fold" and "split" columns
        drugs: drugs dataframe
        extra_datasets: extra datasets to concatenate (e.g. off-label edges, clinical trials)
            Must contain "source", "target" columns
        extra_datasets_names: names of the extra datasets (e.g. "is_off_label", "is_clinical_trial")
            Must be the same length as extra_datasets
    """
    extra_datasets_all_folds = []
    for fold in range(n_folds):
        # Concatenate and label all sources of test pairs
        on_label_fold = splits.filter(pl.col("fold") == fold)
        on_label_fold_test = on_label_fold.filter(pl.col("split") == "TEST")
        on_label_fold_train = on_label_fold.filter(pl.col("split") == "TRAIN")
        on_label_fold_test_positive = on_label_fold_test.filter(pl.col("y") == 1)
        on_label_fold_test_negative = on_label_fold_test.filter(pl.col("y") == 0)
        df_list = [on_label_fold_test_positive, on_label_fold_test_negative] + extra_datasets
        col_name_list = ["is_known_positive", "is_known_negative"] + extra_datasets_names
        full_test_dataset = (
            pl.concat(
                [
                    dataset.with_columns(pl.lit(True).alias(col_name)).select("source", "target", col_name)
                    for dataset, col_name in zip(df_list, col_name_list)
                ],
                how="diagonal",
            )
            .fill_null(False)
            .unique()
        )

        # Generate all drugs x test diseases matrix
        test_dis_all_drugs = (
            drugs.join(full_test_dataset.select("target"), how="cross")
            .join(  # Add ground truth boolean flags
                full_test_dataset, on=["source", "target"], how="left"
            )
            .join(  # Flags for train pairs
                on_label_fold_train.select("source", "target").with_columns(pl.lit(True).alias("is_train")),
                on=["source", "target"],
                how="left",
            )
            .fill_null(False)
            .unique()
        )

        # Filter out train pairs
        test_dis_all_drugs = test_dis_all_drugs.filter(~pl.col("is_train"))
        test_dis_all_drugs = test_dis_all_drugs.drop("is_train")

        # Add back in the ground truth pairs
        extra_datasets_all_folds.append(pl.concat([test_dis_all_drugs, full_test_dataset]))

    return extra_datasets_all_folds


def give_supplemented_matrices(
    matrices_all_folds: List[pl.DataFrame],
    extra_predictions_all_folds: List[pl.DataFrame],
    splits: pl.DataFrame,
    extra_datasets: List[pl.DataFrame],
    extra_datasets_names: List[str],
    perform_checks: bool = True,
    score_col: str = "treat score",
) -> List[pl.DataFrame]:
    """Give supplemented matrices for out of matrix evaluation.

    This is the concatentation of the matrix and the extra datasets for all folds.
    Boolean flags are added to indicate whether a pair is in the matrix, as well as for test pairs.

    Args:
        matrices_all_folds: List of matrix dataframes for all folds
            Must contain "source", "target", score_col columns
        extra_predictions_all_folds: List of extra predictions dataframes for all folds
            Must contain "source", "target", score_col columns
        splits: On-label dataframe
            Must contain "source", "target", "fold" and "split" columns
        extra_datasets: extra datasets to concatenate (e.g. off-label edges, clinical trials)
            Must contain "source", "target" columns
        extra_datasets_names: names of the extra datasets (e.g. "is_off_label", "is_clinical_trial")
            Must be the same length as extra_datasets
        perform_checks: Whether to perform consistency checks on the data
        score_col: Column in the matrix containing the treat scores
    """
    # Get number of folds

    n_folds = len(matrices_all_folds)

    if len(extra_predictions_all_folds) != n_folds:
        raise ValueError("Lengths of matrices_all_folds and extra_predictions_all_folds do not match.")

    # Select required columns

    matrices_all_folds = [
        matrix.select("source", "target", pl.col(score_col).cast(pl.Float32)) for matrix in matrices_all_folds
    ]
    extra_predictions_all_folds = [
        extra_predictions.select("source", "target", pl.col(score_col).cast(pl.Float32))
        for extra_predictions in extra_predictions_all_folds
    ]

    # Concatenate matrices and extra predictions

    supplemented_matrices = []
    for fold in range(3):
        supplemented_matrices.append(
            pl.concat([matrices_all_folds[fold], extra_predictions_all_folds[fold]], how="vertical").unique()
        )

    # Check that extra predictions also in the matrix have consistent scores
    if perform_checks:
        for matrix in supplemented_matrices:
            if len(matrix) != len(matrix.select("source", "target").unique()):
                raise ValueError("Supplied extra predictions has non-matching score in the original matrix")

    # Adding "in_matrix" boolean column

    for fold in range(n_folds):
        supplemented_matrices[fold] = (
            supplemented_matrices[fold]
            .join(
                matrices_all_folds[fold].select("source", "target").with_columns(pl.lit(True).alias("in_matrix")),
                on=["source", "target"],
                how="left",
            )
            .fill_null(False)
        )

    # Adding boolean flags for test pairs

    for fold in range(n_folds):
        on_label_fold_test = splits.filter(pl.col("fold") == fold).filter(pl.col("split") == "TEST")
        on_label_fold_test_positive = on_label_fold_test.filter(pl.col("y") == 1)
        on_label_fold_test_negative = on_label_fold_test.filter(pl.col("y") == 0)
        df_list = [on_label_fold_test_positive, on_label_fold_test_negative] + extra_datasets
        col_name_list = ["is_known_positive", "is_known_negative"] + extra_datasets_names
        full_test_dataset = (
            pl.concat(
                [
                    dataset.with_columns(pl.lit(True).alias(col_name)).select("source", "target", col_name)
                    for dataset, col_name in zip(df_list, col_name_list)
                ],
                how="diagonal",
            )
            .fill_null(False)
            .unique()
        )
        supplemented_matrices[fold] = (
            supplemented_matrices[fold].join(full_test_dataset, on=["source", "target"], how="left").fill_null(False)
        )

    return supplemented_matrices
