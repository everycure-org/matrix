"""
Unit tests for the HFIterableDataset custom Kedro dataset.

These tests focus on testing the custom Kedro dataset implementation
for reading and writing to Hugging Face Hub.
"""

import os
from unittest.mock import Mock, patch

import pytest
from matrix_gcp_datasets.huggingface import (
    HFIterableDataset,
    HFIterableDatasetConfig,
)


def test_config_creation_with_minimal_params():
    """
    GIVEN minimal required parameters (repo_id)
    WHEN HFIterableDatasetConfig is created
    THEN it should use default values for optional parameters
    """
    # Given
    repo_id = "test/repo"

    # When
    config = HFIterableDatasetConfig(repo_id=repo_id)

    # Then
    assert config.repo_id == repo_id
    assert config.split == "train"
    assert config.config_name is None
    assert config.private is False
    assert config.token_key == "HF_TOKEN"
    assert config.token is None
    assert config.dataframe_type == "spark"
    assert config.data_dir is None


def test_config_creation_with_all_params():
    """
    GIVEN all parameters provided
    WHEN HFIterableDatasetConfig is created
    THEN it should use the provided values
    """
    # Given
    repo_id = "test/repo"
    split = "validation"
    config_name = "subset1"
    private = True
    token_key = "CUSTOM_TOKEN"
    token = "secret_token"
    dataframe_type = "pandas"
    data_dir = "/custom/path"

    # When
    config = HFIterableDatasetConfig(
        repo_id=repo_id,
        split=split,
        config_name=config_name,
        private=private,
        token_key=token_key,
        token=token,
        dataframe_type=dataframe_type,
        data_dir=data_dir,
    )

    # Then
    assert config.repo_id == repo_id
    assert config.split == split
    assert config.config_name == config_name
    assert config.private == private
    assert config.token_key == token_key
    assert config.token == token
    assert config.dataframe_type == dataframe_type
    assert config.data_dir == data_dir


def test_config_validation_invalid_dataframe_type():
    """
    GIVEN an invalid dataframe_type
    WHEN HFIterableDatasetConfig is created
    THEN it should raise a ValidationError
    """
    # Given
    repo_id = "test/repo"
    invalid_dataframe_type = "invalid_type"

    # When & Then
    with pytest.raises(Exception):  # ValidationError from pydantic
        HFIterableDatasetConfig(repo_id=repo_id, dataframe_type=invalid_dataframe_type)


def test_build_push_kwargs_with_token():
    """
    GIVEN a config with token
    WHEN build_push_kwargs is called
    THEN it should return correct kwargs
    """
    # Given
    config = HFIterableDatasetConfig(
        repo_id="test/repo", split="train", config_name="subset1", private=True, data_dir="/custom/path"
    )
    token = "test_token"

    # When
    kwargs = config.build_push_kwargs(token)

    # Then
    expected_kwargs = {
        "repo_id": "test/repo",
        "token": "test_token",
        "split": "train",
        "config_name": "subset1",
        "private": True,
        "data_dir": "/custom/path",
    }
    assert kwargs == expected_kwargs


def test_dataset_initialization_with_minimal_params():
    """
    GIVEN minimal parameters
    WHEN HFIterableDataset is initialized
    THEN it should create a valid dataset instance
    """
    # Given
    repo_id = "test/repo"

    # When
    dataset = HFIterableDataset(repo_id=repo_id)

    # Then
    assert isinstance(dataset, HFIterableDataset)
    assert dataset.config.repo_id == repo_id
    assert dataset._credentials is None


def test_dataset_initialization_with_credentials():
    """
    GIVEN credentials parameter
    WHEN HFIterableDataset is initialized
    THEN it should store credentials for token resolution
    """
    # Given
    repo_id = "test/repo"
    credentials = {"HF_TOKEN": "secret_token"}

    # When
    dataset = HFIterableDataset(repo_id=repo_id, credentials=credentials)

    # Then
    assert dataset._credentials == credentials


def test_dataset_initialization_with_invalid_config():
    """
    GIVEN invalid configuration parameters
    WHEN HFIterableDataset is initialized
    THEN it should raise a ValueError
    """
    # Given
    repo_id = "test/repo"
    invalid_dataframe_type = "invalid_type"

    # When & Then
    with pytest.raises(ValueError, match="Invalid HFIterableDataset configuration"):
        HFIterableDataset(repo_id=repo_id, dataframe_type=invalid_dataframe_type)


def test_describe_method():
    """
    GIVEN a HFIterableDataset instance
    WHEN _describe is called
    THEN it should return config information
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo")

    # When
    description = dataset._describe()

    # Then
    assert "config" in description
    assert isinstance(description["config"], str)
    assert "test/repo" in description["config"]


def test_exists_method():
    """
    GIVEN a HFIterableDataset instance
    WHEN _exists is called
    THEN it should always return False (write-only dataset)
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo")

    # When
    exists = dataset._exists()

    # Then
    assert exists is False


@patch.dict(os.environ, {"HF_TOKEN": "env_token"})
def test_resolve_token_from_environment():
    """
    GIVEN HF_TOKEN environment variable is set
    WHEN _resolve_token is called
    THEN it should return the environment token
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo")

    # When
    token = dataset._resolve_token()

    # Then
    assert token == "env_token"


def test_resolve_token_from_credentials():
    """
    GIVEN credentials with token
    WHEN _resolve_token is called
    THEN it should return the credential token
    """
    # Given
    credentials = {"HF_TOKEN": "credential_token"}
    dataset = HFIterableDataset(repo_id="test/repo", credentials=credentials)

    # When
    token = dataset._resolve_token()

    # Then
    assert token == "credential_token"


def test_resolve_token_from_explicit_token():
    """
    GIVEN explicit token parameter
    WHEN _resolve_token is called
    THEN it should return the explicit token (highest priority)
    """
    # Given
    credentials = {"HF_TOKEN": "credential_token"}
    dataset = HFIterableDataset(repo_id="test/repo", credentials=credentials, token="explicit_token")

    # When
    token = dataset._resolve_token()

    # Then
    assert token == "explicit_token"


def test_resolve_token_no_token_available():
    """
    GIVEN no token available from any source
    WHEN _resolve_token is called
    THEN it should raise a ValueError
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo")

    # When & Then
    with pytest.raises(ValueError, match="Hugging Face token not found"):
        dataset._resolve_token()


@patch("datasets.load_dataset")
def test_load_spark_dataframe(mock_load_dataset):
    """
    GIVEN a HFIterableDataset configured for Spark
    WHEN _load is called
    THEN it should load and convert to Spark DataFrame
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="spark", token="test_token")
    mock_hf_dataset = Mock()
    mock_load_dataset.return_value = mock_hf_dataset

    # Mock Spark session and DataFrame creation
    with patch("pyspark.sql.SparkSession") as mock_spark_session:
        mock_spark = Mock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark
        mock_spark_df = Mock()
        mock_spark.createDataFrame.return_value = mock_spark_df

        # Mock Arrow table conversion
        mock_arrow_table = Mock()
        mock_hf_dataset.to_table.return_value = mock_arrow_table

        # When
        result = dataset._load()

        # Then
        mock_load_dataset.assert_called_once()
        mock_hf_dataset.to_table.assert_called_once()
        mock_spark.createDataFrame.assert_called_once_with(mock_arrow_table)
        assert result == mock_spark_df


@patch("datasets.load_dataset")
def test_load_pandas_dataframe(mock_load_dataset):
    """
    GIVEN a HFIterableDataset configured for pandas
    WHEN _load is called
    THEN it should load and convert to pandas DataFrame
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="pandas", token="test_token")
    mock_hf_dataset = Mock()
    mock_load_dataset.return_value = mock_hf_dataset
    expected_pandas_df = Mock()
    mock_hf_dataset.to_pandas.return_value = expected_pandas_df

    # When
    result = dataset._load()

    # Then
    mock_load_dataset.assert_called_once()
    mock_hf_dataset.to_pandas.assert_called_once()
    assert result == expected_pandas_df


@patch("datasets.load_dataset")
def test_load_polars_dataframe(mock_load_dataset):
    """
    GIVEN a HFIterableDataset configured for polars
    WHEN _load is called
    THEN it should load and convert to polars DataFrame
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="polars", token="test_token")
    mock_hf_dataset = Mock()
    mock_load_dataset.return_value = mock_hf_dataset
    expected_polars_df = Mock()
    mock_hf_dataset.to_polars.return_value = expected_polars_df

    # When
    result = dataset._load()

    # Then
    mock_load_dataset.assert_called_once()
    mock_hf_dataset.to_polars.assert_called_once()
    assert result == expected_polars_df


@patch("datasets.Dataset")
def test_save_spark_dataframe(mock_dataset_class):
    """
    GIVEN a Spark DataFrame and HFIterableDataset configured for Spark
    WHEN _save is called
    THEN it should create HF dataset and push to hub
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="spark", token="test_token")
    mock_spark_df = Mock()
    mock_hf_dataset = Mock()
    mock_dataset_class.from_spark.return_value = mock_hf_dataset

    # When
    dataset._save(mock_spark_df)

    # Then
    mock_dataset_class.from_spark.assert_called_once_with(mock_spark_df)
    mock_hf_dataset.push_to_hub.assert_called_once()


@patch("datasets.Dataset")
def test_save_pandas_dataframe(mock_dataset_class):
    """
    GIVEN a pandas DataFrame and HFIterableDataset configured for pandas
    WHEN _save is called
    THEN it should create HF dataset and push to hub
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="pandas", token="test_token")
    mock_pandas_df = Mock()
    mock_hf_dataset = Mock()
    mock_dataset_class.from_pandas.return_value = mock_hf_dataset

    # When
    dataset._save(mock_pandas_df)

    # Then
    mock_dataset_class.from_pandas.assert_called_once_with(mock_pandas_df)
    mock_hf_dataset.push_to_hub.assert_called_once()


@patch("datasets.Dataset")
def test_save_polars_dataframe(mock_dataset_class):
    """
    GIVEN a polars DataFrame and HFIterableDataset configured for polars
    WHEN _save is called
    THEN it should create HF dataset and push to hub
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="polars", token="test_token")
    # Create a proper mock polars DataFrame
    mock_polars_df = Mock()
    mock_hf_dataset = Mock()
    mock_dataset_class.from_polars.return_value = mock_hf_dataset

    # Mock the isinstance check for polars DataFrame
    with patch("huggingface_dataset_demo.hf_kedro.datasets.hf_iterable_dataset.isinstance") as mock_isinstance:
        mock_isinstance.return_value = True

        # When
        dataset._save(mock_polars_df)

        # Then
        mock_polars_df.to_arrow.assert_not_called()  # Should use from_polars directly
        mock_dataset_class.from_polars.assert_called_once_with(mock_polars_df)
        mock_hf_dataset.push_to_hub.assert_called_once()


def test_save_with_invalid_dataframe_type():
    """
    GIVEN invalid dataframe type configuration
    WHEN _save is called
    THEN it should raise a ValueError
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="spark", token="test_token")
    # Manually set invalid dataframe type to bypass pydantic validation
    dataset.config.dataframe_type = "invalid_type"

    # When & Then
    with pytest.raises(ValueError, match="Unsupported dataframe_type"):
        dataset._save(Mock())


def test_error_handling_spark_not_available():
    """
    GIVEN Spark is not installed but dataframe_type='spark'
    WHEN _load is called
    THEN it should raise a RuntimeError
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="spark", token="test_token")

    # When & Then
    with patch("datasets.load_dataset") as mock_load_dataset:
        mock_hf_dataset = Mock()
        mock_load_dataset.return_value = mock_hf_dataset

        with patch("pyspark.sql.SparkSession") as mock_spark_session:
            mock_spark_session.builder.getOrCreate.side_effect = Exception("Spark not available")
            with pytest.raises(Exception, match="Spark not available"):
                dataset._load()


def test_error_handling_polars_not_available():
    """
    GIVEN Polars is not installed but dataframe_type='polars'
    WHEN _load is called
    THEN it should raise a RuntimeError
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="polars", token="test_token")

    # When & Then
    with patch("datasets.load_dataset") as mock_load_dataset:
        mock_hf_dataset = Mock()
        mock_load_dataset.return_value = mock_hf_dataset
        mock_hf_dataset.to_polars.side_effect = ImportError("polars not available")

        with pytest.raises(RuntimeError, match="Polars not available for conversion"):
            dataset._load()


def test_error_handling_pandas_not_available():
    """
    GIVEN pandas is not installed but dataframe_type='pandas'
    WHEN _save is called
    THEN it should raise a TypeError
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="pandas", token="test_token")

    # When & Then
    with patch("datasets.Dataset") as mock_dataset_class:
        mock_dataset_class.from_pandas.side_effect = ImportError("pandas not available")

        with pytest.raises(
            TypeError, match="Configured dataframe_type='pandas' but the input is not a pandas DataFrame"
        ):
            dataset._save(Mock())


@patch("datasets.load_dataset")
def test_arrow_fallback_for_spark(mock_load_dataset):
    """
    GIVEN Spark version < 4.0 and Arrow conversion fails
    WHEN _load is called with dataframe_type='spark'
    THEN it should fallback to pandas conversion
    """
    # Given
    dataset = HFIterableDataset(repo_id="test/repo", dataframe_type="spark", token="test_token")
    mock_hf_dataset = Mock()
    mock_load_dataset.return_value = mock_hf_dataset

    # Mock Spark session
    with patch("pyspark.sql.SparkSession") as mock_spark_session:
        mock_spark = Mock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark
        mock_spark_df = Mock()

        # Mock Arrow table conversion failure
        mock_hf_dataset.to_table.side_effect = Exception("Arrow not available")
        mock_hf_dataset.to_pandas.return_value = Mock()
        mock_spark.createDataFrame.return_value = mock_spark_df

        # When
        result = dataset._load()

        # Then
        mock_hf_dataset.to_table.assert_called_once()
        mock_hf_dataset.to_pandas.assert_called_once()
        mock_spark.createDataFrame.assert_called_once()
        assert result == mock_spark_df
