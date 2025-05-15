import pyspark.sql as ps
import pytest
from matrix.pipelines.matrix_transformations.transformations import AlmostPureRankBasedFrequentFlyerTransformation
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_matrix(spark: SparkSession):
    """Fixture that provides sample drugs data for testing."""
    return spark.createDataFrame(
        data=[
            ("drug_1", "disease_1", 0.9, 0.25),
            ("drug_1", "disease_2", 0.8, 0.5),
            ("drug_2", "disease_1", 0.2, 0.75),
            ("drug_2", "disease_2", 0.1, 1.0),
        ],
        schema=["source", "target", "treat score", "quantile_rank"],
    )


def test_frequent_flyer_transformation(spark, sample_matrix):
    """
    Given a sample matrix, when the frequent flyer transformation is applied,
    then the transformed matrix should be returned.
    """
    # Given sample matrix
    matrix = sample_matrix

    # When the frequent flyer transformation is applied
    result = AlmostPureRankBasedFrequentFlyerTransformation(
        decay=0.05,
        score_col="treat score",
        perform_sort=True,
    ).apply(matrix)

    # Round the transformed score to 3 decimal places
    result = result.withColumn("treat score", F.round(F.col("treat score"), 3))

    # Then the transformed matrix should be returned
    expected = spark.createDataFrame(
        data=[
            ("drug_1", "disease_1", 2.072, 0.25, 1, 0.5, 1, 0.5, 0.9, 1),
            ("drug_1", "disease_2", 2.036, 0.5, 2, 1.0, 1, 0.5, 0.8, 2),
            ("drug_2", "disease_1", 2.036, 0.75, 1, 0.5, 2, 1.0, 0.2, 3),
            ("drug_2", "disease_2", 2.001, 1.0, 2, 1.0, 2, 1.0, 0.1, 4),
        ],
        schema=StructType(
            [
                StructField("source", StringType(), True),
                StructField("target", StringType(), True),
                StructField("treat score", DoubleType(), True),
                StructField("quantile_rank", DoubleType(), True),
                StructField("rank_drug", IntegerType(), False),
                StructField("quantile_drug", DoubleType(), True),
                StructField("rank_disease", IntegerType(), False),
                StructField("quantile_disease", DoubleType(), True),
                StructField("untransformed_treat score", DoubleType(), True),
                StructField("rank", IntegerType(), False),
            ]
        ),
    )

    assertDataFrameEqual(result, expected)
