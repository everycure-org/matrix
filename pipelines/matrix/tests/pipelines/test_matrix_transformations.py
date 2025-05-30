import pytest
from matrix.pipelines.matrix_transformations.transformations import (
    AlmostPureRankBasedFrequentFlyerTransformation,
    NoTransformation,
    RankBasedFrequentFlyerTransformation,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_matrix(spark: SparkSession):
    """Fixture that provides sample drugs data for testing."""
    return spark.createDataFrame(
        data=[
            ("drug_1", "disease_1", 0.9, 0.25, 1),
            ("drug_1", "disease_2", 0.8, 0.5, 2),
            ("drug_2", "disease_1", 0.2, 0.75, 3),
            ("drug_2", "disease_2", 0.1, 1.0, 4),
        ],
        schema=["source", "target", "treat score", "quantile_rank", "rank"],
    )


def test_rank_based_frequent_flyer_transformation(spark, sample_matrix):
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
    result = result.withColumn("treat score", F.round(F.col("treat score"), 3))

    # Then the transformed matrix should be returned with correct weighted scores
    data = [
        {
            "source": "drug_1",
            "target": "disease_1",
            # 2.645 = 0.5 * (0.25) ^ -0.1 + 1.0 * (0.5) ^ -0.05 + 1.0 * (0.5) ^ -0.05
            "treat score": 2.645,
            "quantile_rank": 0.25,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat score": 0.9,
            "rank": 1,
            "untransformed_rank": 1,
        },
        {
            "source": "drug_1",
            "target": "disease_2",
            "treat score": 2.571,
            "quantile_rank": 0.5,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat score": 0.8,
            "rank": 2,
            "untransformed_rank": 2,
        },
        {
            "source": "drug_2",
            "target": "disease_1",
            "treat score": 2.55,
            "quantile_rank": 0.75,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat score": 0.2,
            "rank": 3,
            "untransformed_rank": 3,
        },
        {
            "source": "drug_2",
            "target": "disease_2",
            "treat score": 2.5,
            "quantile_rank": 1.0,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat score": 0.1,
            "rank": 4,
            "untransformed_rank": 4,
        },
    ]

    schema = StructType(
        [
            StructField("source", StringType(), True),
            StructField("target", StringType(), True),
            StructField("treat score", DoubleType(), True),
            StructField("quantile_rank", DoubleType(), True),
            StructField("rank", IntegerType(), False),
            StructField("rank_drug", IntegerType(), False),
            StructField("quantile_drug", DoubleType(), True),
            StructField("rank_disease", IntegerType(), False),
            StructField("quantile_disease", DoubleType(), True),
            StructField("untransformed_treat score", DoubleType(), True),
            StructField("untransformed_rank", IntegerType(), False),
        ]
    )

    expected = spark.createDataFrame(data, schema)

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
    result = result.withColumn("treat score", F.round(F.col("treat score"), 3))

    # Then the transformed matrix should be returned
    data = [
        {
            "source": "drug_1",
            "target": "disease_1",
            # 2.072 = 0.001 * (0.25) ^ -0.05 + 1.0 * (0.5) ^ -0.05 + 1.0 * (0.5) ^ -0.05
            "treat score": 2.072,
            "quantile_rank": 0.25,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat score": 0.9,
            "rank": 1,
            "untransformed_rank": 1,
        },
        {
            "source": "drug_1",
            "target": "disease_2",
            "treat score": 2.036,
            "quantile_rank": 0.5,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 1,
            "quantile_disease": 0.5,
            "untransformed_treat score": 0.8,
            "rank": 2,
            "untransformed_rank": 2,
        },
        {
            "source": "drug_2",
            "target": "disease_1",
            "treat score": 2.036,
            "quantile_rank": 0.75,
            "rank_drug": 1,
            "quantile_drug": 0.5,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat score": 0.2,
            "rank": 3,
            "untransformed_rank": 3,
        },
        {
            "source": "drug_2",
            "target": "disease_2",
            "treat score": 2.001,
            "quantile_rank": 1.0,
            "rank_drug": 2,
            "quantile_drug": 1.0,
            "rank_disease": 2,
            "quantile_disease": 1.0,
            "untransformed_treat score": 0.1,
            "rank": 4,
            "untransformed_rank": 4,
        },
    ]

    schema = StructType(
        [
            StructField("source", StringType(), True),
            StructField("target", StringType(), True),
            StructField("treat score", DoubleType(), True),
            StructField("quantile_rank", DoubleType(), True),
            StructField("rank", IntegerType(), False),
            StructField("rank_drug", IntegerType(), False),
            StructField("quantile_drug", DoubleType(), True),
            StructField("rank_disease", IntegerType(), False),
            StructField("quantile_disease", DoubleType(), True),
            StructField("untransformed_treat score", DoubleType(), True),
            StructField("untransformed_rank", IntegerType(), False),
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
