# NOTE: This file was partially generated using AI assistance.

import pytest
import pyspark.sql as ps
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual

from review_list.pipelines.review_list.nodes import combine_ranked_pair_dataframes, weighted_merge_multiple


@pytest.fixture
def sample_df1(spark):
    """Create first sample dataframe with ranked pairs."""
    schema = StructType([
        StructField("source", StringType(), False),
        StructField("target", StringType(), False),
        StructField("rank", IntegerType(), False),
    ])
    data = [
        ("drug1", "disease1", 1),
        ("drug2", "disease2", 2),
        ("drug3", "disease3", 3),
        ("drug4", "disease4", 4),
        ("drug5", "disease5", 5),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_df2(spark):
    """Create second sample dataframe with ranked pairs."""
    schema = StructType([
        StructField("source", StringType(), False),
        StructField("target", StringType(), False),
        StructField("rank", IntegerType(), False),
    ])
    data = [
        ("drug6", "disease6", 1),
        ("drug7", "disease7", 2),
        ("drug8", "disease8", 3),
        ("drug9", "disease9", 4),
        ("drug10", "disease10", 5),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_df3(spark):
    """Create third sample dataframe with ranked pairs and some duplicates."""
    schema = StructType([
        StructField("source", StringType(), False),
        StructField("target", StringType(), False),
        StructField("rank", IntegerType(), False),
    ])
    data = [
        ("drug1", "disease1", 1),  # Duplicate with df1
        ("drug11", "disease11", 2),
        ("drug12", "disease12", 3),
    ]
    return spark.createDataFrame(data, schema)


@pytest.mark.spark(
    help="This test relies on PYSPARK_PYTHON to be set appropriately, and sometimes does not work in VSCode"
)
class TestWeightedMergeMultiple:
    """Test suite for weighted_merge_multiple function following GivenWhenThen format."""

    def test_weighted_merge_two_dataframes_equal_weights(self, spark, sample_df1, sample_df2):
        """
        Given: Two dataframes with equal weights
        When: Merging with limit 4 and equal weights (0.5, 0.5)
        Then: Should return 2 rows from each dataframe with sequential ranks
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.5), (sample_df2, 0.5)]
        limit = 4

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        assert result.count() == 4
        # Check that we have 2 rows from each source pattern
        df1_rows = result.filter(result.source.startswith("drug")).filter(
            result.source.isin(["drug1", "drug2", "drug3", "drug4", "drug5"])
        ).count()
        df2_rows = result.filter(result.source.startswith("drug")).filter(
            result.source.isin(["drug6", "drug7", "drug8", "drug9", "drug10"])
        ).count()
        assert df1_rows == 2
        assert df2_rows == 2
        
        # Check that ranks are sequential
        ranks = [row.rank for row in result.orderBy("rank").collect()]
        assert ranks == [1, 2, 3, 4]

    def test_weighted_merge_unequal_weights(self, spark, sample_df1, sample_df2):
        """
        Given: Two dataframes with unequal weights
        When: Merging with limit 10 and weights (0.7, 0.3)
        Then: Should return approximately 7 rows from first df and 3 from second
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.7), (sample_df2, 0.3)]
        limit = 10

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        assert result.count() == 8  # Limited by actual data: 5 from df1 + 3 from df2
        # With 70/30 split on limit 10: quota would be 7 and 3, but df1 only has 5 rows
        df1_count = result.filter(result.source.isin(["drug1", "drug2", "drug3", "drug4", "drug5"])).count()
        df2_count = result.filter(result.source.isin(["drug6", "drug7", "drug8", "drug9", "drug10"])).count()
        assert df1_count == 5  # Limited by actual data in df1 (wanted 7, got 5)
        assert df2_count == 3  # Got full quota from df2

    def test_weighted_merge_with_duplicates(self, spark, sample_df1, sample_df3):
        """
        Given: Two dataframes with overlapping data
        When: Merging with duplicates present
        Then: Should deduplicate and maintain proper ranking
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.5), (sample_df3, 0.5)]
        limit = 10

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        # Should have deduplicated the (drug1, disease1) pair
        drug1_count = result.filter((result.source == "drug1") & (result.target == "disease1")).count()
        assert drug1_count == 1
        
        # Check sequential ranking
        ranks = [row.rank for row in result.orderBy("rank").collect()]
        expected_ranks = list(range(1, result.count() + 1))
        assert ranks == expected_ranks

    def test_weighted_merge_single_dataframe(self, spark, sample_df1):
        """
        Given: Single dataframe
        When: Merging with single dataframe
        Then: Should return the dataframe as-is with rewritten ranks
        """
        # Given
        dfs_with_weights = [(sample_df1, 1.0)]
        limit = 10

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        assert result.count() == 5  # All rows from sample_df1
        ranks = [row.rank for row in result.orderBy("rank").collect()]
        assert ranks == [1, 2, 3, 4, 5]

    def test_weighted_merge_weight_normalization(self, spark, sample_df1, sample_df2):
        """
        Given: Dataframes with weights that don't sum to 1
        When: Merging with non-normalized weights
        Then: Should normalize weights and distribute accordingly
        """
        # Given - weights sum to 2.0, should be normalized to 0.5 each
        dfs_with_weights = [(sample_df1, 1.0), (sample_df2, 1.0)]
        limit = 4

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        assert result.count() == 4
        # Should behave same as equal weights after normalization
        df1_rows = result.filter(result.source.isin(["drug1", "drug2", "drug3", "drug4", "drug5"])).count()
        df2_rows = result.filter(result.source.isin(["drug6", "drug7", "drug8", "drug9", "drug10"])).count()
        assert df1_rows == 2
        assert df2_rows == 2

    def test_weighted_merge_zero_quota_handling(self, spark, sample_df1, sample_df2):
        """
        Given: Small limit that results in zero quota for some dataframes
        When: Merging with very small limit
        Then: Should handle zero quotas gracefully
        """
        # Given
        dfs_with_weights = [(sample_df1, 0.1), (sample_df2, 0.9)]  # 10% vs 90%
        limit = 1

        # When
        result = weighted_merge_multiple(dfs_with_weights, limit)

        # Then
        assert result.count() == 1
        # Should get 1 row total, likely from df2 due to higher weight


@pytest.mark.spark(
    help="This test relies on PYSPARK_PYTHON to be set appropriately, and sometimes does not work in VSCode"
)
class TestCombineRankedPairDataframes:
    """Test suite for the main combine_ranked_pair_dataframes function."""

    def test_combine_with_config_parameters(self, spark, sample_df1, sample_df2):
        """
        Given: Valid weights config and dataframes
        When: Calling combine_ranked_pair_dataframes with proper inputs
        Then: Should successfully combine dataframes using configured weights and limit
        """
        # Given
        weights = {
            "input_dataframe_1": {"weight": 0.6},
            "input_dataframe_2": {"weight": 0.4}
        }
        config = {"limit": 6}

        # When
        result = combine_ranked_pair_dataframes(
            weights=weights,
            config=config,
            input_dataframe_1=sample_df1,
            input_dataframe_2=sample_df2
        )

        # Then
        assert isinstance(result, ps.DataFrame)
        assert result.count() <= 6  # Should respect the limit
        
        # Check that ranks are sequential
        ranks = [row.rank for row in result.orderBy("rank").collect()]
        expected_ranks = list(range(1, result.count() + 1))
        assert ranks == expected_ranks

    def test_combine_single_dataframe(self, spark, sample_df1):
        """
        Given: Single dataframe input
        When: Calling combine_ranked_pair_dataframes with one dataframe
        Then: Should return the single dataframe unchanged
        """
        # Given
        weights = {"input_dataframe_1": {"weight": 1.0}}
        config = {"limit": 10}

        # When
        result = combine_ranked_pair_dataframes(
            weights=weights,
            config=config,
            input_dataframe_1=sample_df1
        )

        # Then
        assert isinstance(result, ps.DataFrame)
        assert result.count() == 5  # All rows from sample_df1

    def test_combine_with_default_weights(self, spark, sample_df1, sample_df2):
        """
        Given: Config without explicit weights
        When: Calling combine_ranked_pair_dataframes with missing weights
        Then: Should use default weight of 1.0 for all dataframes
        """
        # Given - weights dict without weight key
        weights = {
            "input_dataframe_1": {},
            "input_dataframe_2": {}
        }
        config = {"limit": 8}

        # When
        result = combine_ranked_pair_dataframes(
            weights=weights,
            config=config,
            input_dataframe_1=sample_df1,
            input_dataframe_2=sample_df2
        )

        # Then
        assert isinstance(result, ps.DataFrame)
        assert result.count() <= 8

    def test_combine_with_default_limit(self, spark, sample_df1):
        """
        Given: Config without explicit limit
        When: Calling combine_ranked_pair_dataframes with missing limit
        Then: Should use default limit of 1000
        """
        # Given
        weights = {"input_dataframe_1": {"weight": 1.0}}
        config = {}  # No limit specified

        # When
        result = combine_ranked_pair_dataframes(
            weights=weights,
            config=config,
            input_dataframe_1=sample_df1
        )

        # Then
        assert isinstance(result, ps.DataFrame)
        assert result.count() == 5  # Limited by actual data, not default limit

    def test_combine_empty_dataframes_raises_error(self, spark):
        """
        Given: No dataframes provided
        When: Calling combine_ranked_pair_dataframes with no dataframes
        Then: Should raise ValueError
        """
        # Given
        weights = {}
        config = {"limit": 10}

        # When/Then
        with pytest.raises(ValueError, match="At least one DataFrame must be provided"):
            combine_ranked_pair_dataframes(weights=weights, config=config)
