import pytest
from matrix.pipelines.matrix_transformations.nodes import return_predictions
from matrix.pipelines.matrix_transformations.transformations import (
    AlmostPureRankBasedFrequentFlyerTransformation,
    NoTransformation,
    RankBasedFrequentFlyerTransformation,
    UniformRankBasedFrequentFlyerTransformation,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_matrix(spark: SparkSession):
    """Fixture that provides sample matrix data for testing."""
    return spark.createDataFrame(
        data=[
            ("drug_a", "disease_a", 0.9, 0.25, 1),
            ("drug_a", "disease_b", 0.8, 0.5, 2),
            ("drug_b", "disease_a", 0.2, 0.75, 3),
            ("drug_b", "disease_b", 0.1, 1.0, 4),
        ],
        schema=["source", "target", "treat score", "quantile_rank", "rank"],
    )


@pytest.fixture
def sample_matrix_with_ties(spark: SparkSession):
    """Fixture that provides sample drugs data for testing."""
    return spark.createDataFrame(
        data=[
            ("drug_a", "disease_a", 0.9, 0.25, 1),
            ("drug_a", "disease_b", 0.8, 0.5, 2),
            ("drug_b", "disease_a", 0.2, 0.75, 3),
            ("drug_b", "disease_b", 0.1, 1.0, 4),
        ],
        schema=["source", "target", "treat score", "quantile_rank", "rank"],
    )


@pytest.fixture
def sample_rank_based_transformed_matrix(spark: SparkSession):
    """Fixture that provides sample transformed matrix data for testing."""
    return spark.createDataFrame(
        data=[
            {
                "source": "drug_a",
                "target": "disease_a",
                # 2.645 = 0.5 * (0.25) ^ -0.1 + 1.0 * (0.5) ^ -0.05 + 1.0 * (0.5) ^ -0.05
                "transformed_treat_score": 2.645,
                "quantile_rank": 0.25,
                "rank_drug": 1,
                "quantile_drug": 0.5,
                "rank_disease": 1,
                "quantile_disease": 0.5,
                "untransformed_treat_score": 0.9,
                "rank": 1,
                "untransformed_rank": 1,
            },
            {
                "source": "drug_a",
                "target": "disease_b",
                "transformed_treat_score": 2.571,
                "quantile_rank": 0.5,
                "rank_drug": 1,
                "quantile_drug": 0.5,
                "rank_disease": 2,
                "quantile_disease": 1.0,
                "untransformed_treat_score": 0.8,
                "rank": 2,
                "untransformed_rank": 2,
            },
            {
                "source": "drug_b",
                "target": "disease_a",
                "transformed_treat_score": 2.55,
                "quantile_rank": 0.75,
                "rank_drug": 2,
                "quantile_drug": 1.0,
                "rank_disease": 1,
                "quantile_disease": 0.5,
                "untransformed_treat_score": 0.2,
                "rank": 3,
                "untransformed_rank": 3,
            },
            {
                "source": "drug_b",
                "target": "disease_b",
                "transformed_treat_score": 2.5,
                "quantile_rank": 1.0,
                "rank_drug": 2,
                "quantile_drug": 1.0,
                "rank_disease": 2,
                "quantile_disease": 1.0,
                "untransformed_treat_score": 0.1,
                "rank": 4,
                "untransformed_rank": 4,
            },
        ],
        schema=StructType(
            [
                StructField("source", StringType(), True),
                StructField("target", StringType(), True),
                StructField("untransformed_treat_score", DoubleType(), True),
                StructField("quantile_rank", DoubleType(), True),
                StructField("untransformed_rank", LongType(), True),
                StructField("rank_drug", IntegerType(), False),
                StructField("quantile_drug", DoubleType(), True),
                StructField("rank_disease", IntegerType(), False),
                StructField("quantile_disease", DoubleType(), True),
                StructField("transformed_treat_score", DoubleType(), True),
                StructField("rank", IntegerType(), False),
            ]
        ),
    )


def test_rank_based_frequent_flyer_transformation(spark, sample_matrix, sample_rank_based_transformed_matrix):
    """
    Given a sample matrix, when the rank-based frequent flyer transformation is applied with custom weights,
    then the transformed matrix should be returned with the correct weighted scores.
    """
    # Given sample matrix
    matrix = sample_matrix

    # When the rank-based frequent flyer transformation is applied with custom weights
    result = RankBasedFrequentFlyerTransformation(
        matrix_weight=0.5,
        drug_weight=1.0,
        disease_weight=1.0,
        decay_matrix=0.1,
        decay_drug=0.05,
        decay_disease=0.05,
    ).apply(matrix_df=matrix, score_col="treat score")

    # Round the transformed score to 3 decimal places
    result = result.withColumn("transformed_treat_score", F.round(F.col("transformed_treat_score"), 3))

    # Then the transformed matrix should be returned with correct weighted scores
    expected = sample_rank_based_transformed_matrix

    assertDataFrameEqual(result, expected)


def test_almost_pure_frequent_flyer_transformation(spark, sample_matrix):
    """
    Given a sample matrix, when the frequent flyer transformation is applied,
    then the transformed matrix should be returned.
    """
    # Given sample matrix
    matrix = sample_matrix

    # When the frequent flyer transformation is applied
    result = AlmostPureRankBasedFrequentFlyerTransformation(
        decay=0.05,
    ).apply(matrix_df=matrix, score_col="treat score")

    # Round the transformed score to 3 decimal places
    result = result.withColumn("transformed_treat_score", F.round(F.col("transformed_treat_score"), 3))

    # Then the transformed matrix should be returned
    data = [
        {
            "source": "drug_a",
            "target": "disease_a",
            # 2.072 = 0.001 * (0.25) ^ -0.05 + 1.0 * (0.5) ^ -0.05 + 1.0 * (0.5) ^ -0.05
            "transformed_treat_score": 2.072,
            "quantile_rank": 0.25,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat_score": 0.9,
            "rank": 1,
            "untransformed_rank": 1,
        },
        {
            "source": "drug_a",
            "target": "disease_b",
            "transformed_treat_score": 2.036,
            "quantile_rank": 0.5,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat_score": 0.8,
            "rank": 2,
            "untransformed_rank": 2,
        },
        {
            "source": "drug_b",
            "target": "disease_a",
            "transformed_treat_score": 2.036,
            "quantile_rank": 0.75,
            "quantile_rank": 0.75,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat_score": 0.2,
            "rank": 3,
            "untransformed_rank": 3,
        },
        {
            "source": "drug_b",
            "target": "disease_b",
            "transformed_treat_score": 2.001,
            "quantile_rank": 1.0,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat_score": 0.1,
            "rank": 4,
            "untransformed_rank": 4,
        },
    ]
    schema = StructType(
        [
            StructField("source", StringType(), True),
            StructField("target", StringType(), True),
            StructField("untransformed_treat_score", DoubleType(), True),
            StructField("quantile_rank", DoubleType(), True),
            StructField("untransformed_rank", LongType(), True),
            StructField("rank_drug", IntegerType(), False),
            StructField("quantile_drug", DoubleType(), True),
            StructField("rank_disease", IntegerType(), False),
            StructField("quantile_disease", DoubleType(), True),
            StructField("transformed_treat_score", DoubleType(), True),
            StructField("rank", IntegerType(), False),
        ]
    )

    expected = spark.createDataFrame(data, schema)

    assertDataFrameEqual(result, expected)


def test_uniform_rank_based_frequent_flyer_transformation(spark, sample_matrix):
    """
    Given a sample matrix, when the frequent flyer transformation is applied,
    then the transformed matrix should be returned.
    """
    # Given sample matrix
    matrix = sample_matrix

    # When the frequent flyer transformation is applied
    result = UniformRankBasedFrequentFlyerTransformation(
        decay=0.05,
    ).apply(matrix_df=matrix, score_col="treat score")

    # Round the transformed score to 3 decimal places
    result = result.withColumn("transformed_treat_score", F.round(F.col("transformed_treat_score"), 3))

    # Then the transformed matrix should be returned
    data = [
        {
            "source": "drug_a",
            "target": "disease_a",
            # 3.142 = 1 * (0.25) ^ -0.05 + 1.0 * (0.5) ^ -0.05 + 1.0 * (0.5) ^ -0.05
            "transformed_treat_score": 3.142,
            "quantile_rank": 0.25,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat_score": 0.9,
            "rank": 1,
            "untransformed_rank": 1,
        },
        {
            "source": "drug_a",
            "target": "disease_b",
            # 3.071 = 1.0 * (0.5) ^ -0.05 + 1.0 * (1.0) ^ -0.05 + 1.0 * (0.5) ^ -0.05
            "transformed_treat_score": 3.071,
            "quantile_rank": 0.5,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat_score": 0.8,
            "rank": 2,
            "untransformed_rank": 2,
        },
        {
            "source": "drug_b",
            "target": "disease_a",
            # 3.050 = 1 * (0.75) ^ -0.05 + 1.0 * (0.5) ^ -0.05 + 1.0 * (1.0) ^ -0.05
            "transformed_treat_score": 3.050,
            "quantile_rank": 0.75,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat_score": 0.2,
            "rank": 3,
            "untransformed_rank": 3,
        },
        {
            "source": "drug_b",
            "target": "disease_b",
            # 3.000 = 1.0 * (1.0) ^ -0.05 + 1.0 * (1.0) ^ -0.05 + 1.0 * (1.0) ^ -0.05
            "transformed_treat_score": 3.000,
            "quantile_rank": 1.0,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat_score": 0.1,
            "rank": 4,
            "untransformed_rank": 4,
        },
    ]

    schema = StructType(
        [
            StructField("source", StringType(), True),
            StructField("target", StringType(), True),
            StructField("untransformed_treat_score", DoubleType(), True),
            StructField("quantile_rank", DoubleType(), True),
            StructField("untransformed_rank", LongType(), True),
            StructField("rank_drug", IntegerType(), False),
            StructField("quantile_drug", DoubleType(), True),
            StructField("rank_disease", IntegerType(), False),
            StructField("quantile_disease", DoubleType(), True),
            StructField("transformed_treat_score", DoubleType(), True),
            StructField("rank", IntegerType(), False),
        ]
    )

    expected = spark.createDataFrame(data, schema)

    assertDataFrameEqual(result, expected)


def test_no_transformation(spark, sample_matrix):
    """
    Given a sample matrix, when the no transformation is applied,
    then the input matrix should be returned unchanged.
    """
    # Given sample matrix
    matrix = sample_matrix

    # When the no transformation is applied
    result = NoTransformation().apply(matrix_df=matrix, score_col="treat score")

    # Then the input matrix should be returned unchanged
    assertDataFrameEqual(result, matrix)


@pytest.fixture
def sample_known_pairs(spark: SparkSession):
    """Fixture that provides sample known pairs data for testing."""

    return spark.createDataFrame(
        data=[
            # Known positives
            {
                "source": "known_drug_a",
                "target": "known_disease_a",
                "y": 1,
                "fold": 3,
                "split": "TRAIN",
            },
            {
                "source": "known_drug_b",
                "target": "known_disease_b",
                "y": 1,
                "fold": 3,
                "split": "TRAIN",
            },
            # Known negative
            {
                "source": "known_drug_c",
                "target": "known_disease_c",
                "y": 0,
                "fold": 3,
                "split": "TRAIN",
            },
            # Known positive - different fold
            {
                "source": "known_drug_d",
                "target": "known_disease_d",
                "y": 1,
                "fold": 1,
                "split": "TRAIN",
            },
            # Test data from different fold
            {
                "source": "known_drug_e",
                "target": "known_disease_e",
                "y": 1,
                "fold": 1,
                "split": "TEST",
            },
        ]
    )


def test_return_predictions(spark, sample_rank_based_transformed_matrix, sample_known_pairs):
    """
    Given a sample matrix and known pairs, when the return predictions function is applied,
    then the predictions should be returned.
    """
    # Given sample matrix and known pairs
    matrix = sample_rank_based_transformed_matrix
    known_pairs = sample_known_pairs

    # When the return predictions function is applied
    n_cross_val_folds = 3
    result = return_predictions(n_cross_val_folds, matrix, known_pairs)

    # The known_pairs from the final cross-validation fold should be included in the result with no scores, and with the correct is_known_positive and is_known_negative columns
    assert result.count() == 7

    assert result.filter(F.col("is_known_positive") == True).count() == 2
    assert result.filter(F.col("is_known_negative") == True).count() == 1

    assert result.filter(F.col("transformed_treat_score").isNull() == True).count() == 3
    assert result.filter(F.col("rank").isNull() == True).count() == 3
