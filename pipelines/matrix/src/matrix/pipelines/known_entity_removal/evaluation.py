"""Module containing functions for evaluation in the known entity removal pipeline."""

import pandas as pd


def give_proportion_with_bootstrap(series: pd.Series, label: bool) -> tuple[float, float]:
    """Give the proportion of series with a given label, with bootstrap uncertainty estimate.

    Args:
        series: Arbitrary Pandas Series.
        label: Chosen label to count occurrences of in the series.

    Returns:
        Tuple:
            - Proportion of elements with the label.
            - Bootstrap uncertainty estimate of the proportion.
    """
    has_label_series = series == label
    num_elements = len(boolean_series)
    num_elements_with_label = boolean_series.sum().item()
    proportion_with_label = num_elements_with_label / num_elements
    # Standard deviation of the proportion upon sampling N elements with replacement
    # https://en.wikipedia.org/wiki/Binomial_distribution
    std = (proportion_with_label * (1 - proportion_with_label) / num_elements) ** (1 / 2)
    return proportion_with_label, std


def give_retrieval_rate(
    pai: pd.DataFrame, bool_col_name: str, filter_col_name: str = "is_known_entity"
) -> tuple[float, float]:
    """Give the proportion of

    Args:
        preds_with_labels: Pandas DataFrame with boolean filter values in the filter_col_name column.
        bool_col_name: Name of a boolean column defining the subset of pairs to compute the retrieval rate for.

    Returns:
        Tuple containing Retrieval rate and uncertainty estimate
    """
    subset_pairs = preds_with_labels[preds_with_labels[bool_col_name]]
    return give_proportion_with_bootstrap(subset_pairs[filter_col_name], False)


def give_removal_rate(
    preds_with_labels: pd.DataFrame, bool_col_name: str, filter_col_name: str = "KE_removal_filter"
) -> float:
    """Give the removal rate of a method for a given subset of pairs.

    Args:
        preds_with_labels: Pandas DataFrame with boolean filter values in the filter_col_name column.
        bool_col_name: Name of a boolean column defining the subset of pairs to compute the removal rate for.

    Returns:
        Tuple:
            - Removal rate.
            - Bootstrap uncertainty estimate of the removal rate.
    """
    subset_pairs = preds_with_labels[preds_with_labels[bool_col_name]]
    return give_proportion_with_bootstrap(subset_pairs[filter_col_name], True)


def give_projected_proportion(
    preds_with_labels: pd.DataFrame, bool_col_name: str, filter_col_name: str = "KE_removal_filter"
) -> float:
    """Give the projected proportion of pairs with True for a given column if the filter is applied.

    Args:
        preds_with_labels: Pandas DataFrame with boolean filter values in the filter_col_name column.
        bool_col_name: Name of a boolean column to count occurrences of in the allowed pairs.

    Returns:
        Tuple:
            - Projected proportion.
            - Bootstrap uncertainty estimate of the projected proportion.
    """
    allowed_pairs = preds_with_labels[~preds_with_labels[filter_col_name]]
    if len(allowed_pairs) == 0:
        return None, None
    return give_proportion_with_bootstrap(allowed_pairs[bool_col_name], True)
