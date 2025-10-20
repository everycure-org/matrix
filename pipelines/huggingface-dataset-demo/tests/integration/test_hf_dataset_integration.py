"""
Integration tests for the HFIterableDataset custom Kedro dataset.

These tests require HF_TOKEN environment variable and perform actual
interactions with Hugging Face Hub.
"""

import os

import pandas as pd
import polars as pl
import pytest
from huggingface_dataset_demo.hf_kedro.datasets.hf_iterable_dataset import HFIterableDataset


@pytest.mark.integration
@pytest.mark.hf_hub
def test_hf_dataset_write_read_cycle_spark(integration_test_setup):
    """
    GIVEN a Spark DataFrame and HFIterableDataset configured for Spark
    WHEN data is written to and read from Hugging Face Hub
    THEN the round-trip should preserve data integrity
    """
    # Given
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not available")

    repo_config = integration_test_setup

    # Create sample Spark DataFrame
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()

    try:
        sample_data = [
            ("EC-DRUG-1", "biolink:treats", "EC-DISEASE-1", "Edge 1"),
            ("EC-DRUG-2", "biolink:causes", "EC-DISEASE-2", "Edge 2"),
        ]
        spark_df = spark.createDataFrame(sample_data, ["subject", "predicate", "object", "description"])

        # Create test repository with subfolder
        test_repo = f"{repo_config['repo_id']}/spark"
        dataset = HFIterableDataset(
            repo_id=test_repo, dataframe_type="spark", token=os.getenv("HF_TOKEN"), private=repo_config["private"]
        )

        # When
        dataset._save(spark_df)
        loaded_df = dataset._load()

        # Then
        assert loaded_df.count() == 2
        assert loaded_df.columns == ["subject", "predicate", "object", "description"]

    finally:
        spark.stop()


@pytest.mark.integration
@pytest.mark.hf_hub
def test_hf_dataset_write_read_cycle_pandas(integration_test_setup):
    """
    GIVEN a pandas DataFrame and HFIterableDataset configured for pandas
    WHEN data is written to and read from Hugging Face Hub
    THEN the round-trip should preserve data integrity
    """
    # Given
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not available")

    repo_config = integration_test_setup

    sample_data = pd.DataFrame(
        {
            "subject": ["EC-DRUG-1", "EC-DRUG-2"],
            "predicate": ["biolink:treats", "biolink:causes"],
            "object": ["EC-DISEASE-1", "EC-DISEASE-2"],
            "description": ["Edge 1", "Edge 2"],
        }
    )

    # Create test repository with subfolder
    test_repo = f"{repo_config['repo_id']}/pandas"
    dataset = HFIterableDataset(
        repo_id=test_repo, dataframe_type="pandas", token=os.getenv("HF_TOKEN"), private=repo_config["private"]
    )

    # When
    dataset._save(sample_data)
    loaded_df = dataset._load()

    # Then
    pd.testing.assert_frame_equal(loaded_df, sample_data)


@pytest.mark.integration
@pytest.mark.hf_hub
def test_hf_dataset_write_read_cycle_polars(integration_test_setup):
    """
    GIVEN a polars DataFrame and HFIterableDataset configured for polars
    WHEN data is written to and read from Hugging Face Hub
    THEN the round-trip should preserve data integrity
    """
    # Given
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not available")

    repo_config = integration_test_setup

    sample_data = pl.DataFrame(
        {
            "subject": ["EC-DRUG-1", "EC-DRUG-2"],
            "predicate": ["biolink:treats", "biolink:causes"],
            "object": ["EC-DISEASE-1", "EC-DISEASE-2"],
            "description": ["Edge 1", "Edge 2"],
        }
    )

    # Create test repository with subfolder
    test_repo = f"{repo_config['repo_id']}/polars"
    dataset = HFIterableDataset(
        repo_id=test_repo, dataframe_type="polars", token=os.getenv("HF_TOKEN"), private=repo_config["private"]
    )

    # When
    dataset._save(sample_data)
    loaded_df = dataset._load()

    # Then
    assert loaded_df.equals(sample_data)


@pytest.mark.integration
@pytest.mark.hf_hub
def test_data_consistency_across_formats(integration_test_setup):
    """
    GIVEN the same data written in different formats
    WHEN read back from HF Hub
    THEN all formats should contain identical data
    """
    # Given
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not available")

    repo_config = integration_test_setup

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "subject": ["EC-DRUG-1", "EC-DRUG-2"],
            "predicate": ["biolink:treats", "biolink:causes"],
            "object": ["EC-DISEASE-1", "EC-DISEASE-2"],
            "description": ["Edge 1", "Edge 2"],
        }
    )

    # Create Spark DataFrame
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()

    try:
        spark_df = spark.createDataFrame(sample_data)
        polars_df = pl.from_pandas(sample_data)

        # Create test repositories with subfolders
        spark_repo = f"{repo_config['repo_id']}/consistency-spark"
        pandas_repo = f"{repo_config['repo_id']}/consistency-pandas"
        polars_repo = f"{repo_config['repo_id']}/consistency-polars"

        # Create datasets
        spark_dataset = HFIterableDataset(
            repo_id=spark_repo, dataframe_type="spark", token=os.getenv("HF_TOKEN"), private=repo_config["private"]
        )
        pandas_dataset = HFIterableDataset(
            repo_id=pandas_repo, dataframe_type="pandas", token=os.getenv("HF_TOKEN"), private=repo_config["private"]
        )
        polars_dataset = HFIterableDataset(
            repo_id=polars_repo, dataframe_type="polars", token=os.getenv("HF_TOKEN"), private=repo_config["private"]
        )

        # When
        spark_dataset._save(spark_df)
        pandas_dataset._save(sample_data)
        polars_dataset._save(polars_df)

        # Read back
        loaded_spark = spark_dataset._load()
        loaded_pandas = pandas_dataset._load()
        loaded_polars = polars_dataset._load()

        # Then
        # Convert all to pandas for comparison
        spark_pandas = loaded_spark.toPandas()
        polars_pandas = loaded_polars.to_pandas()

        pd.testing.assert_frame_equal(spark_pandas, sample_data)
        pd.testing.assert_frame_equal(loaded_pandas, sample_data)
        pd.testing.assert_frame_equal(polars_pandas, sample_data)

    finally:
        spark.stop()
