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
        expected_result = [
            ("df1_drug1", "df1_disease1", 1),
            ("df1&df3_drug5", "df1&df3_disease5", 2),
            ("df1_drug2", "df1_disease2", 3),
            ("df3_drug11", "df3_disease11", 4),
            ("df3_drug12", "df3_disease12", 5),
        ]

        assertDataFrameEqual(result, expected_result)


#     def test_weighted_merge_single_dataframe(self, spark, sample_df1):
#         """
#         Given: Single dataframe
#         When: Merging with single dataframe
#         Then: Should return the dataframe as-is with rewritten ranks
#         """
#         # Given
#         dfs_with_weights = [(sample_df1, 1.0)]
#         limit = 5

#         # When
#         result = weighted_merge_multiple(dfs_with_weights, limit)

#         # Then
#         assert result.count() == 5  # All rows from sample_df1
#         ranks = [row.rank for row in result.orderBy("rank").collect()]
#         assert ranks == [1, 2, 3, 4, 5]

#     # TODO: same as first one
#     def test_weighted_merge_weight_normalization(self, spark, sample_df1, sample_df2):
#         """
#         Given: Dataframes with weights that don't sum to 1
#         When: Merging with non-normalized weights
#         Then: Should normalize weights and distribute accordingly
#         """
#         # Given - weights sum to 2.0, should be normalized to 0.5 each
#         dfs_with_weights = [(sample_df1, 1.0), (sample_df2, 1.0)]
#         limit = 5

#         # When
#         result = weighted_merge_multiple(dfs_with_weights, limit)

#         # Then
#         assert result.count() == 5
#         # Should behave same as equal weights after normalization
#         df1_rows = result.filter(
#             result.source.isin(["df1_drug1", "df1_drug2", "df1_drug3", "df1_drug4", "df1_drug5"])
#         ).count()
#         df2_rows = result.filter(
#             result.source.isin(["df2_drug6", "df2_drug7", "df2_drug8", "df2_drug9", "df2_drug10"])
#         ).count()
#         assert df1_rows + df2_rows == 5
#         assert df1_rows in [2, 3]  # Due to rounding
#         assert df2_rows in [2, 3]

#     def test_weighted_merge_quota_exceeds_available_data(self, spark, sample_df1, sample_df2):
#         """
#         Given: Large limit that exceeds available data in dataframes
#         When: Merging with limit 15 and weights (0.7, 0.3)
#         Then: Should use all available data and show warning
#         """
#         # Given
#         dfs_with_weights = [(sample_df1, 0.7), (sample_df2, 0.3)]
#         limit = 15  # Larger than available data (5 + 5 = 10)

#         # When
#         result = weighted_merge_multiple(dfs_with_weights, limit)

#         # TODO: write out actual result

#         # Then
#         assert result.count() == 10  # All available data (5 + 5)
#         # With 70/30 split on limit 15: quotas would be ~11 and ~4
#         # But df1 only has 5 rows, df2 only has 5 rows
#         df1_count = result.filter(
#             result.source.isin(["df1_drug1", "df1_drug2", "df1_drug3", "df1_drug4", "df1_drug5"])
#         ).count()
#         df2_count = result.filter(
#             result.source.isin(["df2_drug6", "df2_drug7", "df2_drug8", "df2_drug9", "df2_drug10"])
#         ).count()
#         assert df1_count == 5  # All available from df1 (wanted ~11, got 5)
#         assert df2_count == 4   # Limited by quota (wanted ~4, got 4)


# @pytest.mark.spark(
#     help="This test relies on PYSPARK_PYTHON to be set appropriately, and sometimes does not work in VSCode"
# )
# class TestCombineRankedPairDataframes:
#     """Test suite for the main combine_ranked_pair_dataframes function."""

#     def test_combine_with_config_parameters(self, spark, sample_df1, sample_df2):
#         """
#         Given: Valid weights config and dataframes
#         When: Calling combine_ranked_pair_dataframes with proper inputs
#         Then: Should successfully combine dataframes using configured weights and limit
#         """
#         # Given
#         weights = {
#             "input_dataframe_1": {"weight": 0.6},
#             "input_dataframe_2": {"weight": 0.4},
#         }
#         config = {"limit": 5}

#         # When
#         result = combine_ranked_pair_dataframes(
#             weights=weights,
#             config=config,
#             input_dataframe_1=sample_df1,
#             input_dataframe_2=sample_df2,
#         )

#         # Then
#         assert isinstance(result, ps.DataFrame)
#         assert result.count() <= 5  # Should respect the limit

#         # Check that ranks are sequential
#         ranks = [row.rank for row in result.orderBy("rank").collect()]
#         expected_ranks = list(range(1, result.count() + 1))
#         assert ranks == expected_ranks

#     def test_combine_single_dataframe(self, spark, sample_df1):
#         """
#         Given: Single dataframe input
#         When: Calling combine_ranked_pair_dataframes with one dataframe
#         Then: Should return the single dataframe unchanged
#         """
#         # Given
#         weights = {"input_dataframe_1": {"weight": 1.0}}
#         config = {"limit": 10}

#         # When
#         result = combine_ranked_pair_dataframes(
#             weights=weights, config=config, input_dataframe_1=sample_df1
#         )

#         # Then
#         assert isinstance(result, ps.DataFrame)
#         assert result.count() == 5  # All rows from sample_df1

#     def test_combine_with_default_weights(self, spark, sample_df1, sample_df2):
#         """
#         Given: Config without explicit weights
#         When: Calling combine_ranked_pair_dataframes with missing weights
#         Then: Should use default weight of 1.0 for all dataframes
#         """
#         # Given - weights dict without weight key
#         weights = {"input_dataframe_1": {}, "input_dataframe_2": {}}
#         config = {"limit": 5}

#         # When
#         result = combine_ranked_pair_dataframes(
#             weights=weights,
#             config=config,
#             input_dataframe_1=sample_df1,
#             input_dataframe_2=sample_df2,
#         )

#         # Then
#         assert isinstance(result, ps.DataFrame)
#         assert result.count() <= 8

#     def test_combine_with_default_limit(self, spark, sample_df1):
#         """
#         Given: Config without explicit limit
#         When: Calling combine_ranked_pair_dataframes with missing limit
#         Then: Should use default limit of 1000
#         """
#         # Given
#         weights = {"input_dataframe_1": {"weight": 1.0}}
#         config = {}  # No limit specified

#         # When
#         result = combine_ranked_pair_dataframes(
#             weights=weights, config=config, input_dataframe_1=sample_df1
#         )

#         # Then
#         assert isinstance(result, ps.DataFrame)
#         assert result.count() == 5  # Limited by actual data, not default limit

#     def test_combine_empty_dataframes_raises_error(self, spark):
#         """
#         Given: No dataframes provided
#         When: Calling combine_ranked_pair_dataframes with no dataframes
#         Then: Should raise ValueError
#         """
#         # Given
#         weights = {}
#         config = {"limit": 10}

#         # When/Then
#         with pytest.raises(ValueError, match="At least one DataFrame must be provided"):
#             combine_ranked_pair_dataframes(weights=weights, config=config)
