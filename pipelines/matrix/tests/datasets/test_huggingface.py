# NOTE: This file was partially generated using AI assistance.

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from matrix.datasets.huggingface import (
    AuthenticationError,
    HuggingFaceCSVDataset,
    HuggingFaceDatasetError,
    HuggingFaceFileNotFoundError,
    HuggingFaceJSONDataset,
    HuggingFaceParquetDataset,
    HuggingFaceRepositoryNotFoundError,
    HuggingFaceXetDataset,
)


class TestHuggingFaceBaseDataset:
    """Test cases for HuggingFace base dataset functionality."""

    def test_authentication_with_token_in_credentials(self):
        """Test authentication using token in credentials."""
        with patch("matrix.datasets.huggingface.login") as mock_login:
            dataset = HuggingFaceParquetDataset(
                repo_id="test/repo",
                filename="test.parquet",
                credentials={"token": "test_token"}
            )
            
            mock_login.assert_called_once_with(token="test_token")
            assert dataset._token == "test_token"

    def test_authentication_with_token_file(self):
        """Test authentication using token file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("file_token")
            tmp_file.flush()
            
            try:
                with patch("matrix.datasets.huggingface.login") as mock_login:
                    dataset = HuggingFaceParquetDataset(
                        repo_id="test/repo",
                        filename="test.parquet",
                        credentials={"token_file": tmp_file.name}
                    )
                    
                    mock_login.assert_called_once_with(token="file_token")
                    assert dataset._token == "file_token"
            finally:
                os.unlink(tmp_file.name)

    def test_authentication_with_environment_variable(self):
        """Test authentication using environment variable."""
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            with patch("matrix.datasets.huggingface.login") as mock_login:
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet"
                )
                
                mock_login.assert_called_once_with(token="env_token")
                assert dataset._token == "env_token"

    def test_authentication_failure(self):
        """Test authentication failure handling."""
        with patch("matrix.datasets.huggingface.login", side_effect=Exception("Auth failed")):
            with pytest.raises(AuthenticationError, match="Failed to authenticate"):
                HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet",
                    credentials={"token": "invalid_token"}
                )

    def test_no_authentication_anonymous_access(self):
        """Test fallback to anonymous access when no token provided."""
        with patch("matrix.datasets.huggingface.login") as mock_login:
            dataset = HuggingFaceParquetDataset(
                repo_id="test/repo",
                filename="test.parquet"
            )
            
            mock_login.assert_not_called()
            assert dataset._token is None

    def test_exists_file_found(self):
        """Test exists method when file is found."""
        with patch("matrix.datasets.huggingface.login"):
            with patch.object(HuggingFaceParquetDataset, "_api") as mock_api:
                mock_api.hf_hub_download.return_value = "path/to/file"
                
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet"
                )
                
                assert dataset.exists() is True

    def test_exists_file_not_found(self):
        """Test exists method when file is not found."""
        with patch("matrix.datasets.huggingface.login"):
            with patch.object(HuggingFaceParquetDataset, "_api") as mock_api:
                mock_api.hf_hub_download.side_effect = HfHubHTTPError("Not found")
                
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet"
                )
                
                assert dataset.exists() is False

    def test_describe_method(self):
        """Test dataset description method."""
        with patch("matrix.datasets.huggingface.login"):
            dataset = HuggingFaceParquetDataset(
                repo_id="test/repo",
                filename="test.parquet",
                revision="v1.0"
            )
            
            description = dataset._describe()
            
            assert description["repo_id"] == "test/repo"
            assert description["filename"] == "test.parquet"
            assert description["revision"] == "v1.0"
            assert description["type"] == "HuggingFaceParquetDataset"


class TestHuggingFaceParquetDataset:
    """Test cases for HuggingFace Parquet dataset."""

    def test_load_parquet_file(self):
        """Test loading a Parquet file from HuggingFace Hub."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10.0, 20.0, 30.0]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            test_df.to_parquet(tmp_file.name)
            
            try:
                with patch("matrix.datasets.huggingface.login"):
                    with patch("matrix.datasets.huggingface.hf_hub_download", return_value=tmp_file.name):
                        dataset = HuggingFaceParquetDataset(
                            repo_id="test/repo",
                            filename="test.parquet"
                        )
                        
                        loaded_df = dataset.load()
                        
                        pd.testing.assert_frame_equal(loaded_df, test_df)
            finally:
                os.unlink(tmp_file.name)

    def test_save_parquet_file(self):
        """Test saving a Parquet file to HuggingFace Hub."""
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10.0, 20.0, 30.0]
        })
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.upload_file") as mock_upload:
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet",
                    save_args={
                        "commit_message": "Test upload",
                        "commit_description": "Test description"
                    }
                )
                
                dataset.save(test_df)
                
                # Verify upload was called
                mock_upload.assert_called_once()
                call_args = mock_upload.call_args
                
                assert call_args.kwargs["path_in_repo"] == "test.parquet"
                assert call_args.kwargs["repo_id"] == "test/repo"
                assert call_args.kwargs["commit_message"] == "Test upload"
                assert call_args.kwargs["commit_description"] == "Test description"

    def test_load_with_pandas_args(self):
        """Test loading with custom pandas arguments."""
        test_df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            test_df.to_parquet(tmp_file.name)
            
            try:
                with patch("matrix.datasets.huggingface.login"):
                    with patch("matrix.datasets.huggingface.hf_hub_download", return_value=tmp_file.name):
                        dataset = HuggingFaceParquetDataset(
                            repo_id="test/repo",
                            filename="test.parquet",
                            load_args={
                                "pandas_args": {
                                    "columns": ["col1", "col2"]
                                }
                            }
                        )
                        
                        loaded_df = dataset.load()
                        
                        assert list(loaded_df.columns) == ["col1", "col2"]
                        assert len(loaded_df) == 3
            finally:
                os.unlink(tmp_file.name)


class TestHuggingFaceCSVDataset:
    """Test cases for HuggingFace CSV dataset."""

    def test_load_csv_file(self):
        """Test loading a CSV file from HuggingFace Hub."""
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10.0, 20.0, 30.0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp_file:
            test_df.to_csv(tmp_file.name, index=False)
            
            try:
                with patch("matrix.datasets.huggingface.login"):
                    with patch("matrix.datasets.huggingface.hf_hub_download", return_value=tmp_file.name):
                        dataset = HuggingFaceCSVDataset(
                            repo_id="test/repo",
                            filename="test.csv"
                        )
                        
                        loaded_df = dataset.load()
                        
                        pd.testing.assert_frame_equal(loaded_df, test_df)
            finally:
                os.unlink(tmp_file.name)

    def test_save_csv_file(self):
        """Test saving a CSV file to HuggingFace Hub."""
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"]
        })
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.upload_file") as mock_upload:
                dataset = HuggingFaceCSVDataset(
                    repo_id="test/repo",
                    filename="test.csv"
                )
                
                dataset.save(test_df)
                
                mock_upload.assert_called_once()


class TestHuggingFaceJSONDataset:
    """Test cases for HuggingFace JSON dataset."""

    def test_load_json_file(self):
        """Test loading a JSON file from HuggingFace Hub."""
        test_data = {
            "nodes": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"}
            ],
            "metadata": {"version": "1.0"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_file:
            json.dump(test_data, tmp_file)
            
            try:
                with patch("matrix.datasets.huggingface.login"):
                    with patch("matrix.datasets.huggingface.hf_hub_download", return_value=tmp_file.name):
                        dataset = HuggingFaceJSONDataset(
                            repo_id="test/repo",
                            filename="test.json"
                        )
                        
                        loaded_data = dataset.load()
                        
                        assert loaded_data == test_data
            finally:
                os.unlink(tmp_file.name)

    def test_save_json_file(self):
        """Test saving a JSON file to HuggingFace Hub."""
        test_data = {"key": "value", "number": 42}
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.upload_file") as mock_upload:
                dataset = HuggingFaceJSONDataset(
                    repo_id="test/repo",
                    filename="test.json"
                )
                
                dataset.save(test_data)
                
                mock_upload.assert_called_once()


class TestHuggingFaceXetDataset:
    """Test cases for HuggingFace Xet dataset."""

    def test_xet_initialization(self):
        """Test Xet dataset initialization with custom arguments."""
        with patch("matrix.datasets.huggingface.login"):
            dataset = HuggingFaceXetDataset(
                repo_id="test/repo",
                filename="large_file.xet",
                xet_args={
                    "chunk_size": "128MB",
                    "compression": "gzip"
                },
                load_args={"lazy": True}
            )
            
            assert dataset._chunk_size == "128MB"
            assert dataset._compression == "gzip"
            assert dataset._lazy_loading is True

    def test_xet_load_parquet_fallback(self):
        """Test Xet dataset loading with Parquet fallback."""
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "data": ["x", "y", "z"]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            test_df.to_parquet(tmp_file.name)
            
            try:
                with patch("matrix.datasets.huggingface.login"):
                    with patch("matrix.datasets.huggingface.hf_hub_download", return_value=tmp_file.name):
                        dataset = HuggingFaceXetDataset(
                            repo_id="test/repo",
                            filename="test.xet"
                        )
                        
                        loaded_data = dataset.load()
                        
                        pd.testing.assert_frame_equal(loaded_data, test_df)
            finally:
                os.unlink(tmp_file.name)

    def test_xet_save_dataframe(self):
        """Test Xet dataset saving DataFrame."""
        test_df = pd.DataFrame({"col": [1, 2, 3]})
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.upload_file") as mock_upload:
                dataset = HuggingFaceXetDataset(
                    repo_id="test/repo",
                    filename="test.xet"
                )
                
                dataset.save(test_df)
                
                mock_upload.assert_called_once()

    def test_xet_save_bytes(self):
        """Test Xet dataset saving binary data."""
        test_bytes = b"binary data content"
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.upload_file") as mock_upload:
                dataset = HuggingFaceXetDataset(
                    repo_id="test/repo",
                    filename="test.xet"
                )
                
                dataset.save(test_bytes)
                
                mock_upload.assert_called_once()

    def test_xet_save_unsupported_type(self):
        """Test Xet dataset with unsupported data type."""
        with patch("matrix.datasets.huggingface.login"):
            dataset = HuggingFaceXetDataset(
                repo_id="test/repo",
                filename="test.xet"
            )
            
            with pytest.raises(ValueError, match="Unsupported data type"):
                dataset.save({"unsupported": "dict"})


class TestErrorHandling:
    """Test cases for error handling in HuggingFace datasets."""

    def test_repository_not_found_error(self):
        """Test handling of repository not found error."""
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.hf_hub_download", 
                      side_effect=RepositoryNotFoundError("Repo not found")):
                dataset = HuggingFaceParquetDataset(
                    repo_id="nonexistent/repo",
                    filename="test.parquet"
                )
                
                with pytest.raises(HuggingFaceRepositoryNotFoundError, 
                                 match="Repository nonexistent/repo not found"):
                    dataset.load()

    def test_file_not_found_error(self):
        """Test handling of file not found error."""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.hf_hub_download", 
                      side_effect=HfHubHTTPError("File not found", response=mock_response)):
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="nonexistent.parquet"
                )
                
                with pytest.raises(HuggingFaceFileNotFoundError, 
                                 match="File nonexistent.parquet not found"):
                    dataset.load()

    def test_generic_http_error(self):
        """Test handling of generic HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.hf_hub_download", 
                      side_effect=HfHubHTTPError("Server error", response=mock_response)):
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet"
                )
                
                with pytest.raises(HuggingFaceDatasetError, match="HTTP error downloading file"):
                    dataset.load()

    def test_upload_error(self):
        """Test handling of upload errors."""
        test_df = pd.DataFrame({"col": [1, 2, 3]})
        
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.upload_file", 
                      side_effect=Exception("Upload failed")):
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet"
                )
                
                with pytest.raises(HuggingFaceDatasetError, match="Failed to upload file"):
                    dataset.save(test_df)

    def test_retry_mechanism(self):
        """Test retry mechanism for transient errors."""
        with patch("matrix.datasets.huggingface.login"):
            with patch("matrix.datasets.huggingface.hf_hub_download") as mock_download:
                # First two calls fail, third succeeds
                mock_download.side_effect = [
                    ConnectionError("Network error"),
                    TimeoutError("Timeout"),
                    "/path/to/file"
                ]
                
                dataset = HuggingFaceParquetDataset(
                    repo_id="test/repo",
                    filename="test.parquet"
                )
                
                # Should succeed after retries
                result = dataset._download_file()
                assert result == "/path/to/file"
                assert mock_download.call_count == 3


class TestIntegrationScenarios:
    """Integration test scenarios for HuggingFace datasets."""

    def test_full_workflow_parquet(self):
        """Test complete workflow: save and load Parquet dataset."""
        test_df = pd.DataFrame({
            "drug_id": ["DRUG001", "DRUG002", "DRUG003"],
            "disease_id": ["DISEASE001", "DISEASE002", "DISEASE003"],
            "score": [0.85, 0.72, 0.91]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            try:
                with patch("matrix.datasets.huggingface.login"):
                    with patch("matrix.datasets.huggingface.upload_file") as mock_upload:
                        with patch("matrix.datasets.huggingface.hf_hub_download", return_value=tmp_file.name):
                            # Save dataset
                            save_dataset = HuggingFaceParquetDataset(
                                repo_id="everycure/test-dataset",
                                filename="drug_disease_scores.parquet",
                                save_args={
                                    "commit_message": "Add drug-disease scores",
                                    "commit_description": "Initial dataset upload"
                                }
                            )
                            
                            # Simulate saving by creating the file
                            test_df.to_parquet(tmp_file.name)
                            save_dataset.save(test_df)
                            
                            # Load dataset
                            load_dataset = HuggingFaceParquetDataset(
                                repo_id="everycure/test-dataset",
                                filename="drug_disease_scores.parquet"
                            )
                            
                            loaded_df = load_dataset.load()
                            
                            # Verify data integrity
                            pd.testing.assert_frame_equal(loaded_df, test_df)
                            
                            # Verify upload was called with correct parameters
                            mock_upload.assert_called_once()
                            call_kwargs = mock_upload.call_args.kwargs
                            assert call_kwargs["repo_id"] == "everycure/test-dataset"
                            assert call_kwargs["path_in_repo"] == "drug_disease_scores.parquet"
                            assert call_kwargs["commit_message"] == "Add drug-disease scores"
            finally:
                os.unlink(tmp_file.name)

    def test_kedro_catalog_integration(self):
        """Test integration with Kedro catalog configuration."""
        # This would test the dataset in a Kedro catalog context
        # For now, we test the configuration parsing
        
        config = {
            "type": "matrix.datasets.huggingface.HuggingFaceParquetDataset",
            "repo_id": "everycure/matrix-kg-v1.0",
            "filename": "nodes.parquet",
            "revision": "v1.0.0",
            "credentials": "huggingface_token",
            "load_args": {
                "pandas_args": {
                    "columns": ["id", "name", "category"]
                }
            },
            "save_args": {
                "commit_message": "Update knowledge graph nodes",
                "pandas_args": {
                    "compression": "snappy"
                }
            },
            "metadata": {
                "description": "Knowledge graph nodes for drug repurposing",
                "tags": ["knowledge-graph", "drug-repurposing"]
            }
        }
        
        with patch("matrix.datasets.huggingface.login"):
            dataset = HuggingFaceParquetDataset(
                repo_id=config["repo_id"],
                filename=config["filename"],
                revision=config["revision"],
                load_args=config["load_args"],
                save_args=config["save_args"],
                metadata=config["metadata"]
            )
            
            assert dataset._repo_id == "everycure/matrix-kg-v1.0"
            assert dataset._filename == "nodes.parquet"
            assert dataset._revision == "v1.0.0"
            assert dataset._load_args == config["load_args"]
            assert dataset._save_args == config["save_args"]
            assert dataset._metadata == config["metadata"]