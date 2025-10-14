# NOTE: This file was partially generated using AI assistance.

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset, DatasetDict

from matrix.datasets.huggingface import HuggingFaceDataset
from kedro.io.core import DatasetError


class TestHuggingFaceDataset:
    """Test suite for HuggingFaceDataset."""

    def test_init_with_minimal_params(self):
        """Given minimal parameters, when initializing dataset, then it should work correctly."""
        # Given
        repo_id = "test/dataset"
        
        # When
        dataset = HuggingFaceDataset(repo_id=repo_id)
        
        # Then
        assert dataset._repo_id == repo_id
        assert dataset._token is None
        assert dataset._private is False
        assert dataset._save_args == {}
        assert dataset._load_args == {}

    def test_init_with_all_params(self):
        """Given all parameters, when initializing dataset, then it should store them correctly."""
        # Given
        repo_id = "test/dataset"
        token = "hf_test_token"
        private = True
        save_args = {"commit_message": "Test commit"}
        load_args = {"split": "train"}
        
        # When
        dataset = HuggingFaceDataset(
            repo_id=repo_id,
            token=token,
            private=private,
            save_args=save_args,
            load_args=load_args
        )
        
        # Then
        assert dataset._repo_id == repo_id
        assert dataset._token == token
        assert dataset._private is True
        assert dataset._save_args == save_args
        assert dataset._load_args == load_args

    def test_describe(self):
        """Given a dataset, when calling describe, then it should return correct description."""
        # Given
        dataset = HuggingFaceDataset(
            repo_id="test/dataset",
            private=True,
            save_args={"commit_message": "Test"},
            load_args={"split": "train"}
        )
        
        # When
        description = dataset._describe()
        
        # Then
        expected = {
            "repo_id": "test/dataset",
            "private": True,
            "save_args": {"commit_message": "Test"},
            "load_args": {"split": "train"}
        }
        assert description == expected

    @patch('matrix.datasets.huggingface.create_repo')
    @patch('matrix.datasets.huggingface.HfApi')
    def test_ensure_repo_exists_when_repo_exists(self, mock_hf_api, mock_create_repo):
        """Given existing repo, when ensuring repo exists, then it should not create new repo."""
        # Given
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_info.return_value = {"id": "test/dataset"}
        
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        dataset._ensure_repo_exists()
        
        # Then
        mock_api_instance.repo_info.assert_called_once_with("test/dataset", repo_type="dataset")
        mock_create_repo.assert_not_called()

    @patch('matrix.datasets.huggingface.create_repo')
    @patch('matrix.datasets.huggingface.HfApi')
    def test_ensure_repo_exists_when_repo_not_exists(self, mock_hf_api, mock_create_repo):
        """Given non-existing repo, when ensuring repo exists, then it should create new repo."""
        # Given
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_info.side_effect = Exception("Repo not found")
        
        dataset = HuggingFaceDataset(repo_id="test/dataset", token="test_token", private=True)
        
        # When
        dataset._ensure_repo_exists()
        
        # Then
        mock_create_repo.assert_called_once_with(
            repo_id="test/dataset",
            repo_type="dataset",
            private=True,
            token="test_token"
        )

    def test_save_single_dataframe(self):
        """Given a DataFrame, when saving to temp dir, then it should create parquet file."""
        # Given
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30]
        })
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset._save_single_dataframe(df, temp_path)
            
            # Then
            parquet_file = temp_path / "data.parquet"
            assert parquet_file.exists()
            
            # Verify content
            loaded_df = pd.read_parquet(parquet_file)
            pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_multiple_dataframes(self):
        """Given multiple DataFrames, when saving to temp dir, then it should create multiple parquet files."""
        # Given
        nodes_df = pd.DataFrame({"id": ["n1", "n2"], "type": ["drug", "disease"]})
        edges_df = pd.DataFrame({"source": ["n1"], "target": ["n2"], "relation": ["treats"]})
        data_dict = {"nodes": nodes_df, "edges": edges_df}
        
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset._save_multiple_dataframes(data_dict, temp_path)
            
            # Then
            nodes_file = temp_path / "nodes.parquet"
            edges_file = temp_path / "edges.parquet"
            
            assert nodes_file.exists()
            assert edges_file.exists()
            
            # Verify content
            loaded_nodes = pd.read_parquet(nodes_file)
            loaded_edges = pd.read_parquet(edges_file)
            
            pd.testing.assert_frame_equal(nodes_df, loaded_nodes)
            pd.testing.assert_frame_equal(edges_df, loaded_edges)

    def test_create_dataset_card_single_dataframe(self):
        """Given single DataFrame, when creating dataset card, then it should generate correct README."""
        # Given
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset._create_dataset_card(temp_path, df)
            
            # Then
            readme_file = temp_path / "README.md"
            assert readme_file.exists()
            
            content = readme_file.read_text()
            assert "2 rows, 2 columns" in content
            assert "data.parquet" in content
            assert "test/dataset" in content

    def test_create_dataset_card_multiple_dataframes(self):
        """Given multiple DataFrames, when creating dataset card, then it should generate correct README."""
        # Given
        data_dict = {
            "nodes": pd.DataFrame({"id": [1, 2, 3]}),
            "edges": pd.DataFrame({"source": [1], "target": [2]})
        }
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset._create_dataset_card(temp_path, data_dict)
            
            # Then
            readme_file = temp_path / "README.md"
            assert readme_file.exists()
            
            content = readme_file.read_text()
            assert "nodes.parquet**: 3 rows, 1 columns" in content
            assert "edges.parquet**: 1 rows, 2 columns" in content

    @patch('matrix.datasets.huggingface.upload_folder')
    @patch('matrix.datasets.huggingface.HuggingFaceDataset._ensure_repo_exists')
    def test_save_single_dataframe_success(self, mock_ensure_repo, mock_upload):
        """Given DataFrame, when saving successfully, then it should upload to HF Hub."""
        # Given
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        dataset = HuggingFaceDataset(
            repo_id="test/dataset",
            token="test_token",
            save_args={"commit_message": "Test upload"}
        )
        
        mock_commit_info = Mock()
        mock_commit_info.oid = "abc123"
        mock_upload.return_value = mock_commit_info
        
        # When
        dataset.save(df)
        
        # Then
        mock_ensure_repo.assert_called_once()
        mock_upload.assert_called_once()
        
        # Verify upload_folder arguments
        call_args = mock_upload.call_args
        assert call_args[1]["repo_id"] == "test/dataset"
        assert call_args[1]["repo_type"] == "dataset"
        assert call_args[1]["commit_message"] == "Test upload"
        assert call_args[1]["token"] == "test_token"

    @patch('matrix.datasets.huggingface.upload_folder')
    @patch('matrix.datasets.huggingface.HuggingFaceDataset._ensure_repo_exists')
    def test_save_multiple_dataframes_success(self, mock_ensure_repo, mock_upload):
        """Given multiple DataFrames, when saving successfully, then it should upload to HF Hub."""
        # Given
        data_dict = {
            "nodes": pd.DataFrame({"id": [1, 2]}),
            "edges": pd.DataFrame({"source": [1], "target": [2]})
        }
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        mock_commit_info = Mock()
        mock_commit_info.oid = "abc123"
        mock_upload.return_value = mock_commit_info
        
        # When
        dataset.save(data_dict)
        
        # Then
        mock_ensure_repo.assert_called_once()
        mock_upload.assert_called_once()

    def test_save_invalid_data_type(self):
        """Given invalid data type, when saving, then it should raise DatasetError."""
        # Given
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        invalid_data = "not a dataframe"
        
        # When & Then
        with pytest.raises(DatasetError, match="can only save pandas DataFrames"):
            dataset.save(invalid_data)

    def test_save_invalid_dict_values(self):
        """Given dict with non-DataFrame values, when saving, then it should raise DatasetError."""
        # Given
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        invalid_dict = {"nodes": pd.DataFrame({"id": [1]}), "edges": "not a dataframe"}
        
        # When & Then
        with pytest.raises(DatasetError, match="All values in data dictionary must be pandas DataFrames"):
            dataset.save(invalid_dict)

    @patch('matrix.datasets.huggingface.load_dataset')
    def test_load_single_dataset_success(self, mock_load_dataset):
        """Given single dataset on HF Hub, when loading, then it should return DataFrame."""
        # Given
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.to_pandas.return_value = df
        mock_load_dataset.return_value = mock_dataset
        
        dataset = HuggingFaceDataset(
            repo_id="test/dataset",
            token="test_token",
            load_args={"split": "train", "revision": "main"}
        )
        
        # When
        result = dataset.load()
        
        # Then
        mock_load_dataset.assert_called_once_with(
            "test/dataset",
            split="train",
            revision="main",
            streaming=False,
            token="test_token"
        )
        pd.testing.assert_frame_equal(result, df)

    @patch('matrix.datasets.huggingface.load_dataset')
    def test_load_dataset_dict_success(self, mock_load_dataset):
        """Given DatasetDict on HF Hub, when loading, then it should return dict of DataFrames."""
        # Given
        nodes_df = pd.DataFrame({"id": [1, 2]})
        edges_df = pd.DataFrame({"source": [1], "target": [2]})
        
        mock_nodes_dataset = Mock(spec=Dataset)
        mock_nodes_dataset.to_pandas.return_value = nodes_df
        mock_edges_dataset = Mock(spec=Dataset)
        mock_edges_dataset.to_pandas.return_value = edges_df
        
        mock_dataset_dict = Mock(spec=DatasetDict)
        mock_dataset_dict.items.return_value = [
            ("nodes", mock_nodes_dataset),
            ("edges", mock_edges_dataset)
        ]
        mock_load_dataset.return_value = mock_dataset_dict
        
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        result = dataset.load()
        
        # Then
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        pd.testing.assert_frame_equal(result["nodes"], nodes_df)
        pd.testing.assert_frame_equal(result["edges"], edges_df)

    @patch('matrix.datasets.huggingface.load_dataset')
    def test_load_failure(self, mock_load_dataset):
        """Given load failure, when loading, then it should raise DatasetError."""
        # Given
        mock_load_dataset.side_effect = Exception("Network error")
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When & Then
        with pytest.raises(DatasetError, match="Failed to load dataset from test/dataset"):
            dataset.load()

    @patch('matrix.datasets.huggingface.HfApi')
    def test_exists_when_repo_exists(self, mock_hf_api):
        """Given existing repo, when checking exists, then it should return True."""
        # Given
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_info.return_value = {"id": "test/dataset"}
        
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        result = dataset.exists()
        
        # Then
        assert result is True
        mock_api_instance.repo_info.assert_called_once_with("test/dataset", repo_type="dataset")

    @patch('matrix.datasets.huggingface.HfApi')
    def test_exists_when_repo_not_exists(self, mock_hf_api):
        """Given non-existing repo, when checking exists, then it should return False."""
        # Given
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_info.side_effect = Exception("Repo not found")
        
        dataset = HuggingFaceDataset(repo_id="test/dataset")
        
        # When
        result = dataset.exists()
        
        # Then
        assert result is False