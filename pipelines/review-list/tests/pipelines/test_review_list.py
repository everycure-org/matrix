import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual

from review_list.pipelines.review_list.nodes import (
    weighted_merge_multiple,
)


@pytest.fixture
def sample_df1(spark):
    """Create first sample dataframe with ranked pairs."""
    return spark.createDataFrame(
        [
            ("df1_drug1", "df1_disease1", 1),
            ("df1_drug2", "df1_disease2", 2),
            ("df1_drug3", "df1_disease3", 3),
            ("df1_drug4", "df1_disease4", 4),
            ("df1&df3_drug5", "df1&df3_disease5", 5),
        ],
        schema=StructType(
            [
                StructField("source", StringType(), False),
                StructField("target", StringType(), False),
                StructField("rank", IntegerType(), False),
            ]
        ),
    )


@pytest.fixture
def sample_df2(spark):
    """Create second sample dataframe with ranked pairs."""
    schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("rank", IntegerType(), False),
        ]
    )
    data = [
        ("df2_drug6", "df2_disease6", 1),
        ("df2_drug7", "df2_disease7", 2),
        ("df2_drug8", "df2_disease8", 3),
        ("df2_drug9", "df2_disease9", 4),
        ("df2_drug10", "df1_disease10", 5),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_df3(spark):
    """Create third sample dataframe with ranked pairs and some duplicates."""
    schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("rank", IntegerType(), False),
        ]
    )
    data = [
        ("df1&df3_drug5", "df1&df3_disease5", 1),  # Duplicate with df1
        ("df3_drug11", "df3_disease11", 2),
        ("df3_drug12", "df3_disease12", 3),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_df4(spark):
    """Create fourth sample dataframe with only 2 rows."""
    schema = StructType(
        [
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("rank", IntegerType(), False),
        ]
    )
    data = [
        ("df4_drug13", "df4_disease13", 1),
        ("df4_drug14", "df4_disease14", 2),
    ]
    return spark.createDataFrame(data, schema)


class TestWeightedMergeMultiple:
    """Test suite for weighted_merge_multiple function following GivenWhenThen format."""

    def test_weighted_merge_two_dataframes_equal_weights(
        self, spark, sample_df1, sample_df2
    ):
        """
        Given: Two dataframes with equal weights
        When: Merging with limit 8 and equal weights (0.5, 0.5).
        Then: Should return 3 rows from each dataframe with sequential ranks
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.5), (sample_df2, 0.5)]
        limit = 6

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        expected_result = [
            ("df1_drug1", "df1_disease1", 1),
            ("df2_drug6", "df2_disease6", 2),
            ("df1_drug2", "df1_disease2", 3),
            ("df2_drug7", "df2_disease7", 4),
            ("df1_drug3", "df1_disease3", 5),
            ("df2_drug8", "df2_disease8", 6),
        ]

        assertDataFrameEqual(result, expected_result)

    def test_weighted_merge_unequal_weights(self, spark, sample_df1, sample_df2):
        """
        Given: Two dataframes with unequal weights
        When: Merging with limit 5 and weights (0.8, 0.2)
        Then: Should return 4 rows from first df and 1 from second
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.8), (sample_df2, 0.2)]
        limit = 5

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        expected_result = [
            ("df1_drug1", "df1_disease1", 1),
            ("df2_drug6", "df2_disease6", 2),
            ("df1_drug2", "df1_disease2", 3),
            ("df1_drug3", "df1_disease3", 4),
            ("df1_drug4", "df1_disease4", 5),
        ]

        assertDataFrameEqual(result, expected_result)

    def test_weighted_merge_with_duplicates(self, spark, sample_df1, sample_df3):
        """
        Given: Two dataframes with overlapping data
        When: Merging with duplicates present
        Then: Should deduplicate and maintain proper ranking
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.5), (sample_df3, 0.5)]
        limit = 5

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        # expected_result = [
        #     ("df1_drug1", "df1_disease1", 1),
        #     ("df1&df3_drug5", "df1&df3_disease5", 2),
        #     ("df1_drug2", "df1_disease2", 3),
        #     ("df3_drug11", "df3_disease11", 4),
        #     ("df3_drug12", "df3_disease12", 5),
        # ]

        # TODO: update this if we re-calculate quotas post deduplication
        expected_result = [
            ("df1_drug1", "df1_disease1", 1),
            ("df1&df3_drug5", "df1&df3_disease5", 2),
            ("df1_drug2", "df1_disease2", 3),
            ("df3_drug11", "df3_disease11", 4),
            ("df1_drug3", "df1_disease3", 5),
        ]

        assertDataFrameEqual(result, expected_result)

    def test_weighted_merge_single_dataframe(self, spark, sample_df1):
        """
        Given: Single dataframe
        When: Merging with single dataframe
        Then: Should return the dataframe as-is with rewritten ranks
        """
        # Given
        dfs_with_weights = [(sample_df1, 1.0)]
        limit = 5

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        assertDataFrameEqual(result, sample_df1)

    def test_weighted_merge_weight_normalization(self, spark, sample_df1, sample_df2):
        """
        Given: Dataframes with weights that don't produce integer quotas (e.g., 0.7 and 0.3 with limit 5)
        When: Merging with weights that result in fractional quotas (3.5 and 1.5)
        Then: Should round quotas appropriately and distribute accordingly
        """
        dfs_with_weights = [(sample_df1, 0.7), (sample_df2, 0.3)]
        limit = 5

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        # With 70/30 weights and limit 5: quotas are 3.5 and 1.5, rounded to 3 and 2
        expected_result = [
            ("df1_drug1", "df1_disease1", 1),
            ("df2_drug6", "df2_disease6", 2),
            ("df1_drug2", "df1_disease2", 3),
            ("df2_drug7", "df2_disease7", 4),
            ("df1_drug3", "df1_disease3", 5),
        ]
        assertDataFrameEqual(result, expected_result)

    # TODO: fix this one
    def test_weighted_merge_quota_exceeds_available_data(
        self, spark, sample_df1, sample_df2
    ):
        """
        Given: Large limit that exceeds available data in dataframes
        When: Merging with limit 15 and weights (0.7, 0.3)
        Then: Should use all available data and show warning
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.5), (sample_df4, 0.5)]
        limit = 15  # Larger than available data

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        expected_result = [
            ("df1_drug1", "df1_disease1", 1),
            ("df4_drug13", "df4_disease13", 2),
            ("df1_drug2", "df1_disease2", 3),
            ("df4_drug14", "df4_disease14", 4),
            ("df1_drug2", "df1_disease2", 5),
            ("df1_drug3", "df1_disease3", 6),
            ("df1_drug4", "df1_disease4", 7),
            ("df1&df3_drug5", "df1&df3_disease5", 8),
        ]

        assertDataFrameEqual(result, expected_result)


def test_weighted_merge_weights_sum_to_one(self, spark, sample_df1, sample_df2):
    """
    Given: Dataframes with weights that don't sum to 1
    When: Merging with weights that don't sum to 1
    Then: Should raise an error
    """
    dfs_with_weights = [(sample_df1, 1.0), (sample_df2, 0.6)]
    limit = 5
    with pytest.raises(ValueError) as e:
        weighted_merge_multiple(dfs_with_weights, limit)
    assert str(e.value) == "Weights must sum to 1"
