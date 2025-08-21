import random

import pandas as pd
import pytest
from pyspark.testing import assertDataFrameEqual

from review_list.pipelines.review_list.nodes import (
    prefetch_top_quota,
    weighted_interleave_dataframes,
)


@pytest.fixture
def sample_df1():
    """Create first sample dataframe with ranked pairs."""
    return pd.DataFrame(
        [
            ("df1_drug1", "df1_disease1", 1),
            ("df1_drug2", "df1_disease2", 2),
            ("df1_drug3", "df1_disease3", 3),
            ("df1_drug4", "df1_disease4", 4),
            ("df1&df3_drug5", "df1&df3_disease5", 5),
        ],
        columns=["source", "target", "rank"],
    )


@pytest.fixture
def sample_df2():
    """Create second sample dataframe with ranked pairs."""
    return pd.DataFrame(
        [
            ("df2_drug6", "df2_disease6", 1),
            ("df2_drug7", "df2_disease7", 2),
            ("df2_drug8", "df2_disease8", 3),
            ("df2_drug9", "df2_disease9", 4),
            ("df2_drug10", "df2_disease10", 5),
        ],
        columns=["source", "target", "rank"],
    )


@pytest.fixture
def sample_df3():
    """Create third sample dataframe with ranked pairs and some duplicates."""
    return pd.DataFrame(
        [
            ("df1&df3_drug5", "df1&df3_disease5", 1),  # Duplicate with df1
            ("df3_drug11", "df3_disease11", 2),
            ("df3_drug12", "df3_disease12", 3),
        ],
        columns=["source", "target", "rank"],
    )


@pytest.fixture
def sample_df4():
    """Create fourth sample dataframe with only 2 rows."""
    return pd.DataFrame(
        [
            ("df4_drug13", "df4_disease13", 1),
            ("df4_drug14", "df4_disease14", 2),
        ],
        columns=["source", "target", "rank"],
    )


def test_weighted_interleave_two_dataframes_equal_weights(sample_df1, sample_df2):
    """
    Given: Two dataframes with equal weights
    When: Interleaving with limit 6 and equal weights (0.5, 0.5).
    Then: Should return 6 rows with no duplicates and sequential ranks
    """
    # Given
    weights = {"sample_df1": {"weight": 0.5}, "sample_df2": {"weight": 0.5}}
    config = {"limit": 6}

    # When
    result = weighted_interleave_dataframes(
        weights=weights,
        config=config,
        rng=random.Random(1),
        sample_df1=sample_df1,
        sample_df2=sample_df2,
    )

    # Then
    expected_result = pd.DataFrame(
        [
            ("df1_drug1", "df1_disease1", "sample_df1", 1),
            ("df2_drug6", "df2_disease6", "sample_df2", 2),
            ("df2_drug7", "df2_disease7", "sample_df2", 3),
            ("df1_drug2", "df1_disease2", "sample_df1", 4),
            ("df1_drug3", "df1_disease3", "sample_df1", 5),
            ("df1_drug4", "df1_disease4", "sample_df1", 6),
        ],
        columns=["source", "target", "from_input_datasets", "rank"],
    )

    assert result.equals(expected_result)


def test_weighted_interleave_with_duplicates_across_dataframes(sample_df1, sample_df3):
    """
    Given: Two dataframes with overlapping data (duplicate source-target pairs)
    When: Interleaving with equal weights (0.5, 0.5)
    Then: Should deduplicate and maintain proper ranking
    """
    # Given
    weights = {"sample_df1": {"weight": 0.5}, "sample_df3": {"weight": 0.5}}
    config = {"limit": 8}

    # When
    result = weighted_interleave_dataframes(
        weights,
        config,
        rng=random.Random(42),  # Different seed for different deterministic result
        sample_df1=sample_df1,
        sample_df3=sample_df3,
    )

    # Then
    # The duplicate pair "df1&df3_drug5" should only appear once
    expected_result = pd.DataFrame(
        [
            ("df1&df3_drug5", "df1&df3_disease5", "sample_df3,sample_df1", 1),
            ("df1_drug1", "df1_disease1", "sample_df1", 2),
            ("df1_drug2", "df1_disease2", "sample_df1", 3),
            ("df1_drug3", "df1_disease3", "sample_df1", 4),
            ("df3_drug11", "df3_disease11", "sample_df3", 5),
            ("df3_drug12", "df3_disease12", "sample_df3", 6),
            ("df1_drug4", "df1_disease4", "sample_df1", 7),
        ],
        columns=["source", "target", "from_input_datasets", "rank"],
    )

    assert result.equals(expected_result)


def test_weighted_interleave_unequal_weights(sample_df1, sample_df2):
    """
    Given: Two dataframes with unequal weights
    When: Interleaving with limit 5 and weights (0.7, 0.3)
    Then: Should return 5 rows respecting the weight distribution
    """
    # Given
    weights = {"sample_df1": {"weight": 0.7}, "sample_df2": {"weight": 0.3}}
    config = {"limit": 5}

    # When
    result = weighted_interleave_dataframes(
        weights,
        config,
        rng=random.Random(40),  # Different seed for different deterministic result
        sample_df1=sample_df1,
        sample_df2=sample_df2,
    )

    # Then
    expected_result = pd.DataFrame(
        [
            ("df1_drug1", "df1_disease1", "sample_df1", 1),
            ("df2_drug6", "df2_disease6", "sample_df2", 2),
            ("df1_drug2", "df1_disease2", "sample_df1", 3),
            ("df1_drug3", "df1_disease3", "sample_df1", 4),
            ("df2_drug7", "df2_disease7", "sample_df2", 5),
        ],
        columns=["source", "target", "from_input_datasets", "rank"],
    )

    assert result.equals(expected_result)


def test_weighted_interleave_single_dataframe(sample_df1):
    """
    Given: One dataframe
    When: Interleaving with limit 3
    Then: Should return 3 rows from the single dataframe
    """
    # Given
    weights = {"sample_df1": {"weight": 1.0}}
    config = {"limit": 3}

    # When
    result = weighted_interleave_dataframes(
        weights, config, rng=None, sample_df1=sample_df1
    )

    # Then
    # All rows should come from sample_df1, but with the new column structure
    expected_result = sample_df1.head(3).copy()
    expected_result["from_input_datasets"] = "sample_df1"
    expected_result = expected_result[
        ["source", "target", "from_input_datasets", "rank"]
    ]
    assert result.equals(expected_result)


def test_weighted_interleave_all_four_dataframes(
    sample_df1, sample_df2, sample_df3, sample_df4
):
    """
    Given: Four dataframes with various sizes and overlaps
    When: Interleaving with equal weights among all four
    Then: Should return 'limit' rows from all dataframes, deduplicated with sequential ranks
    """
    # Given
    weights = {
        "sample_df1": {"weight": 0.2},
        "sample_df2": {"weight": 0.3},
        "sample_df3": {"weight": 0.1},
        "sample_df4": {"weight": 0.4},
    }
    config = {"limit": 8}

    # When
    result = weighted_interleave_dataframes(
        weights=weights,
        config=config,
        rng=random.Random(7),
        sample_df1=sample_df1,
        sample_df2=sample_df2,
        sample_df3=sample_df3,
        sample_df4=sample_df4,
    )

    # Then
    expected_result = pd.DataFrame(
        [
            ("df2_drug6", "df2_disease6", "sample_df2", 1),
            ("df1_drug1", "df1_disease1", "sample_df1", 2),
            ("df4_drug13", "df4_disease13", "sample_df4", 3),
            ("df1_drug2", "df1_disease2", "sample_df1", 4),
            # This row was only sampled from df3, not df1, hence why it's not in the from_input_datasets column
            ("df1&df3_drug5", "df1&df3_disease5", "sample_df3", 5),
            ("df2_drug7", "df2_disease7", "sample_df2", 6),
            ("df1_drug3", "df1_disease3", "sample_df1", 7),
            ("df3_drug11", "df3_disease11", "sample_df3", 8),
        ],
        columns=["source", "target", "from_input_datasets", "rank"],
    )

    assert result.equals(expected_result)


def test_weighted_interleave_weights_sum_to_one(sample_df1, sample_df2):
    """
    Given: Dataframes with weights that don't sum to 1
    When: Interleaving with weights that don't sum to 1
    Then: Should raise an error
    """
    # Given
    weights = {"sample_df1": {"weight": 1.0}, "sample_df2": {"weight": 0.6}}
    config = {"limit": 5}

    # When/Then
    with pytest.raises(ValueError) as e:
        weighted_interleave_dataframes(
            weights,
            config,
            rng=random.Random(789),
            sample_df1=sample_df1,
            sample_df2=sample_df2,
        )
    assert str(e.value) == "Weights must sum to 1"


def test_weighted_interleave_limit_exceeds_available_data_warns(
    sample_df1, sample_df2, caplog
):
    """
    Given: Large limit that exceeds available data in dataframes
    When: Interleaving with limit 15 and weights (0.7, 0.3)
    Then: Should use all available data and show warning
    """
    # Given
    total_available = len(sample_df1) + len(sample_df2)
    weights = {
        "sample_df1": {"weight": 0.7},
        "sample_df2": {"weight": 0.3},
    }
    config = {"limit": 15}

    # When
    caplog.clear()
    with caplog.at_level("WARNING"):
        result = weighted_interleave_dataframes(
            weights,
            config,
            rng=random.Random(5),
            sample_df1=sample_df1,
            sample_df2=sample_df2,
        )

    # Then
    assert len(result) == total_available  # we cannot exceed available unique rows
    assert any("Requested limit" in rec.message for rec in caplog.records)


# Tests for prefetch_top_quota function
@pytest.fixture
def sample_spark_df1(spark):
    """Create first sample Spark DataFrame with ranked pairs."""
    return spark.createDataFrame(
        [
            ("df1_drug1", "df1_disease1", 1),
            ("df1_drug2", "df1_disease1", 2),
            ("df1_drug3", "df1_disease1", 3),
            ("df1_drug4", "df1_disease1", 4),
            ("df1_drug5", "df1_disease1", 5),
        ],
        ["source", "target", "rank"],
    )


@pytest.fixture
def sample_spark_df2(spark):
    """Create second sample Spark DataFrame with ranked pairs."""
    return spark.createDataFrame(
        [
            ("df2_drug6", "df2_disease6", 1),
            ("df2_drug7", "df2_disease7", 2),
            ("df2_drug8", "df2_disease8", 3),
            ("df2_drug9", "df2_disease9", 4),
            ("df2_drug10", "df2_disease10", 5),
        ],
        ["source", "target", "rank"],
    )


def test_prefetch_top_quota(spark, sample_spark_df1, sample_spark_df2):
    """
    Given: Two Spark DataFrames with unequal weights (0.7, 0.3)
    When: Prefetching with limit 10
    Then: Should return DataFrames with proportional quotas + 20% buffer
    """
    # Given
    weights = {"sample_spark_df1": {"weight": 0.7}, "sample_spark_df2": {"weight": 0.3}}
    config = {"limit": 5}

    # When
    result = prefetch_top_quota(
        weights=weights,
        config=config,
        sample_spark_df1=sample_spark_df1,
        sample_spark_df2=sample_spark_df2,
    )

    # Then

    expected_result = [
        # First DataFrame: quota = 4, buffer = ceil(4 * 1.2) = 5
        spark.createDataFrame(
            [
                ("df1_drug1", "df1_disease1", 1),
                ("df1_drug2", "df1_disease1", 2),
                ("df1_drug3", "df1_disease1", 3),
                ("df1_drug4", "df1_disease1", 4),
                ("df1_drug5", "df1_disease1", 5),
            ],
            ["source", "target", "rank"],
        ),
        # Second DataFrame: quota = 1.5 -> 2, buffer = ceil(2 * 1.2) = 2.4  -> 3
        spark.createDataFrame(
            [
                ("df2_drug6", "df2_disease6", 1),
                ("df2_drug7", "df2_disease7", 2),
                ("df2_drug8", "df2_disease8", 3),
            ],
            ["source", "target", "rank"],
        ),
    ]

    assertDataFrameEqual(result[0], expected_result[0])
    assertDataFrameEqual(result[1], expected_result[1])


def test_prefetch_top_quota_single_dataframe(spark, sample_spark_df1):
    """
    Given: Single Spark DataFrame with weight 1.0
    When: Prefetching with limit 3
    Then: Should return single DataFrame with quota + 20% buffer
    """
    # Given
    weights = {"sample_spark_df1": {"weight": 1.0}}
    config = {"limit": 3}

    # When
    result = prefetch_top_quota(
        weights=weights, config=config, sample_spark_df1=sample_spark_df1
    )

    # Then
    expected_result = sample_spark_df1.head(4)
    assertDataFrameEqual(result[0], expected_result)


def test_prefetch_top_quota_weights_must_sum_to_one(
    spark, sample_spark_df1, sample_spark_df2
):
    """
    Given: Two Spark DataFrames with weights that don't sum to 1
    When: Prefetching with invalid weights
    Then: Should raise ValueError
    """
    # Given
    weights = {
        "sample_spark_df1": {"weight": 0.6},
        "sample_spark_df2": {"weight": 0.5},  # Sum = 1.1, not 1.0
    }
    config = {"limit": 5}

    # When/Then
    with pytest.raises(ValueError) as e:
        prefetch_top_quota(
            weights=weights,
            config=config,
            sample_spark_df1=sample_spark_df1,
            sample_spark_df2=sample_spark_df2,
        )
    assert str(e.value) == "Weights must sum to 1"


def test_prefetch_top_quota_missing_limit(spark, sample_spark_df1):
    """
    Given: Config without limit
    When: Prefetching without limit
    Then: Should raise ValueError
    """
    # Given
    weights = {"sample_spark_df1": {"weight": 1.0}}
    config = {}  # Missing limit

    # When/Then
    with pytest.raises(ValueError) as e:
        prefetch_top_quota(
            weights=weights, config=config, sample_spark_df1=sample_spark_df1
        )
    assert str(e.value) == "Missing limit in config"


def test_prefetch_top_quota_no_dataframes(spark):
    """
    Given: No Spark DataFrames
    When: Prefetching with no inputs
    Then: Should raise ValueError
    """
    # Given
    weights = {}
    config = {"limit": 5}

    # When/Then
    with pytest.raises(ValueError) as e:
        prefetch_top_quota(weights=weights, config=config)
    assert str(e.value) == "At least one DataFrame must be provided"
