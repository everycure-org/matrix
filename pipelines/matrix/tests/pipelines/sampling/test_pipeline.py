"""
Tests for the sampling pipeline.
"""

import pytest
from pyspark.sql import DataFrame, SparkSession
import random
from matrix.pipelines.sampling.nodes import sample_datasets


def random_dataframe(spark) -> DataFrame:
    # generate 10 randomly named columns with random data (float and strings)
    # and 10-100 rows each
    return spark.createDataFrame(
        [(i, *[random.random() for _ in range(10)]) for i in range(100)],
        ["id", *[f"col_{i}" for i in range(10)]],
    )


@pytest.fixture
def df1(spark: SparkSession) -> DataFrame:
    return random_dataframe(spark)


@pytest.fixture
def df2(spark: SparkSession) -> DataFrame:
    return random_dataframe(spark)


def test_sample_datasets(df1: DataFrame, df2: DataFrame):
    # When
    sampled_dfs = sample_datasets(df1, df2, row_count=2)

    # Then
    assert len(sampled_dfs) == 2
    assert all(df.count() == 2 for df in sampled_dfs)
    # assert all columns are present
    assert all(col in sampled_dfs[0].columns for col in df1.columns)
    assert all(col in sampled_dfs[1].columns for col in df2.columns)
