"""Module containing functions for evaluation in the known entity removal pipeline."""

import pandas as pd


def give_proportion_with_bootstrap(series: pd.Series, label: bool) -> tuple[float, float]:
    """Give the proportion of series with a given label, with bootstrap uncertainty estimate.

    Args:
        series: Arbitrary Pandas Series.
        label: Chosen label to count occurrences of in the series.

    Returns:
        Tuple containing the proportion and a bootstrap uncertainty estimate.
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
    pairs: pd.DataFrame, bool_col_name: str, filter_col_name: str = "is_known_entity"
) -> tuple[float, float]:
    """Give the proportion of a subset of pairs that a boolean filter retains.

    Args:
        pairs: DataFrame with boolean column defining a subset and a boolean column defining the filter.
        bool_col_name: Name of the boolean column defining the subset of pairs.
        filter_col_name: Name of the boolean column defining the subset of pairs.

    Returns:
        Tuple containing the retrieval rate and a bootstrap uncertainty estimate.
    """
    subset_pairs = pairs[pairs[bool_col_name]]
    return give_proportion_with_bootstrap(subset_pairs[filter_col_name], False)


def give_removal_rate(
    preds_with_labels: pd.DataFrame, bool_col_name: str, filter_col_name: str = "KE_removal_filter"
) -> float:
    """Give the proportion of a subset of pairs that a boolean filter removes.

    Args:
        pairs: DataFrame with boolean column defining a subset and a boolean column defining the filter.
        bool_col_name: Name of the boolean column defining the subset of pairs.
        filter_col_name: Name of the boolean column defining the subset of pairs.

    Returns:
        Tuple containing the removal rate and a bootstrap uncertainty estimate.
    """
    subset_pairs = pairs[pairs[bool_col_name]]
    return give_proportion_with_bootstrap(subset_pairs[filter_col_name], True)


def give_projected_proportion(
    pairs: pd.DataFrame, bool_col_name: str, filter_col_name: str = "KE_removal_filter"
) -> float:
    """Give the projected proportion of pairs with True for a given target column if the filter is applied.

    Args:
        pairs: DataFrame with the target boolean column and a boolean column defining the filter.
        bool_col_name: Name of the target boolean column.
        filter_col_name: Name of the boolean column defining the subset of pairs.

    Returns:
        Tuple containing the projected proportion and a bootstrap uncertainty estimate.
    """
    allowed_pairs = preds_with_labels[~preds_with_labels[filter_col_name]]
    if len(allowed_pairs) == 0:
        return None, None
    return give_proportion_with_bootstrap(allowed_pairs[bool_col_name], True)
