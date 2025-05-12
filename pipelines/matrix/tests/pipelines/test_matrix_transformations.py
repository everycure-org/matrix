import pyspark.sql as ps
import pytest
from matrix.pipelines.matrix_transformations.nodes import frequent_flyer_transformation
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def sample_matrix(spark: SparkSession):
    """Fixture that provides sample drugs data for testing."""
    return spark.createDataFrame(
        data=[
            ("drug_1", "disease_1", 0.9),
            ("drug_1", "disease_2", 0.8),
            ("drug_2", "disease_1", 0.2),
            ("drug_2", "disease_2", 0.1),
        ],
        schema=["source", "target", "treat score"],
    )


def test_frequent_flyer_transformation(spark, sample_matrix):
    result = frequent_flyer_transformation(sample_matrix)

    # Round the transformed score to 3 decimal places
    result = result.withColumn("transformed_score", F.round(F.col("transformed_score"), 3))

    result.orderBy("treat score", ascending=False).show()
    # Define the exact schema to match the transformation output
    schema = StructType(
        [
            StructField("source", StringType(), True),
            StructField("target", StringType(), True),
            StructField("treat score", DoubleType(), True),
            StructField("rank_drug", IntegerType(), False),
            StructField("rank_disease", IntegerType(), False),
            StructField("rank_matrix", IntegerType(), False),
            StructField("quantile_drug", DoubleType(), True),
            StructField("quantile_disease", DoubleType(), True),
            StructField("quantile_matrix", DoubleType(), True),
            StructField("transformed_score", DoubleType(), True),
        ]
    )

    # Create expected DataFrame with the exact schema
    expected = spark.createDataFrame(
        data=[
            ("drug_1", "disease_1", 0.9, 1, 1, 1, 0.5, 0.5, 0.25, 2.072),
            ("drug_1", "disease_2", 0.8, 2, 1, 2, 1.0, 0.5, 0.5, 2.036),
            ("drug_2", "disease_1", 0.2, 1, 2, 3, 0.5, 1.0, 0.75, 2.036),
            ("drug_2", "disease_2", 0.1, 2, 2, 4, 1.0, 1.0, 1.0, 2.001),
        ],
        schema=schema,
    )

    # Compare DataFrames
    assertDataFrameEqual(result, expected)
