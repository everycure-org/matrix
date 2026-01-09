import numpy as np
import pandas as pd
from matrix.pipelines.known_entity_removal.evaluation import (
    give_projected_proportion,
    give_proportion_with_bootstrap,
    give_removal_rate,
    give_retrieval_rate,
)


def test_give_proportion_with_bootstrap_all_true():
    # Given a series with all True values
    series = pd.Series([True, True, True, True])

    # When we calculate the proportion with bootstrap
    proportion, std = give_proportion_with_bootstrap(series, True)

    # Then the proportion should be 1.0 and std should be 0.0
    assert proportion == 1.0
    assert std == 0.0


def test_give_proportion_with_bootstrap_all_false():
    # Given a series with all False values
    series = pd.Series([False, False, False])

    # When we calculate the proportion with bootstrap for True label
    proportion, std = give_proportion_with_bootstrap(series, True)

    # Then the proportion should be 0.0 and std should be 0.0
    assert proportion == 0.0
    assert std == 0.0


def test_give_proportion_with_bootstrap_mixed():
    # Given a series with mixed True and False values
    series = pd.Series([True, False, True, False, True])

    # When we calculate the proportion with bootstrap for True label
    proportion, std = give_proportion_with_bootstrap(series, True)

    # Then the proportion should be 0.6 (3/5) and std should be approximately correct
    assert proportion == 0.6
    expected_std = np.sqrt(0.6 * 0.4 / 5)
    assert abs(std - expected_std) < 1e-10


def test_give_proportion_with_bootstrap_empty_series():
    # Given an empty series
    series = pd.Series([], dtype=bool)

    # When we calculate the proportion with bootstrap
    proportion, std = give_proportion_with_bootstrap(series, True)

    # Then both should be None
    assert proportion is None
    assert std is None


def test_give_retrieval_rate():
    # Given a dataframe with a subset column and a filter column
    pairs = pd.DataFrame(
        {
            "subset_col": [True, True, True, False, False],
            "is_known_entity": [False, False, True, False, True],
        }
    )

    # When we calculate the retrieval rate (proportion of subset NOT filtered)
    mean, std = give_retrieval_rate(pairs, "subset_col", "is_known_entity")

    # Then the mean should be 2/3 (2 out of 3 subset pairs are not known entities)
    assert abs(mean - 2 / 3) < 1e-10
    assert std is not None


def test_give_retrieval_rate_empty_subset():
    # Given a dataframe with no rows matching the subset
    pairs = pd.DataFrame(
        {
            "subset_col": [False, False],
            "is_known_entity": [False, True],
        }
    )

    # When we calculate the retrieval rate
    mean, std = give_retrieval_rate(pairs, "subset_col", "is_known_entity")

    # Then both should be None (empty subset)
    assert mean is None
    assert std is None


def test_give_removal_rate():
    # Given a dataframe with a subset column and a filter column
    pairs = pd.DataFrame(
        {
            "subset_col": [True, True, True, False, False],
            "is_known_entity": [True, False, True, False, True],
        }
    )

    # When we calculate the removal rate (proportion of subset that IS filtered)
    mean, std = give_removal_rate(pairs, "subset_col", "is_known_entity")

    # Then the mean should be 2/3 (2 out of 3 subset pairs are known entities)
    assert abs(mean - 2 / 3) < 1e-10
    assert std is not None


def test_give_projected_proportion():
    # Given a dataframe with a target column and a filter column
    pairs = pd.DataFrame(
        {
            "target_col": [True, True, False, True, False],
            "is_known_entity": [True, False, False, True, False],
        }
    )

    # When we calculate the projected proportion (proportion of non-filtered pairs with target True)
    mean, std = give_projected_proportion(pairs, "target_col", "is_known_entity")

    # Then the mean should be 1/3 (1 out of 3 non-filtered pairs have target True)
    assert abs(mean - 1 / 3) < 1e-10
    assert std is not None


def test_give_projected_proportion_all_filtered():
    # Given a dataframe where all pairs are filtered
    pairs = pd.DataFrame(
        {
            "target_col": [True, True, False],
            "is_known_entity": [True, True, True],
        }
    )

    # When we calculate the projected proportion
    mean, std = give_projected_proportion(pairs, "target_col", "is_known_entity")

    # Then both should be None (no non-filtered pairs)
    assert mean is None
    assert std is None
