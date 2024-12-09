import pytest
from unittest.mock import patch, MagicMock
from pipelines.matrix.src.matrix.datasets.graph import PandasParquetDataset
from kedro.io.core import DatasetError
import pandas as pd

from matrix.datasets.graph import KnowledgeGraphDataset, KnowledgeGraph


def test_load_success():
    with patch(
        "pipelines.matrix.src.matrix.datasets.graph.ParquetDataset._load", return_value=MagicMock()
    ) as mock_load:
        dataset = PandasParquetDataset(filepath="dummy_path")
        result = dataset._load()

        mock_load.assert_called_once()
        assert result is mock_load.return_value


def test_load_with_as_type():
    with patch(
        "pipelines.matrix.src.matrix.datasets.graph.ParquetDataset._load", return_value=MagicMock()
    ) as mock_load:
        dataset = PandasParquetDataset(filepath="dummy_path", load_args={"as_type": "float32"})
        dataset._load()

        mock_load.assert_called_once()
        assert dataset._as_type == "float32"


def test_load_file_not_found():
    with patch("pipelines.matrix.src.matrix.datasets.graph.ParquetDataset._load", side_effect=FileNotFoundError):
        dataset = PandasParquetDataset(filepath="dummy_path")

        with pytest.raises(DatasetError, match="Unable to find the Parquet file `dummy_path` underlying this dataset!"):
            dataset._load()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": ["drug1", "disease1"],
            "is_drug": [True, False],
            "is_disease": [False, True],
            "topological_embedding": [[1.0, 2.0], [3.0, 4.0]],
        }
    )


@pytest.fixture
def mock_parquet_file(tmp_path, sample_df):
    file_path = tmp_path / "test_graph.parquet"
    sample_df.to_parquet(file_path)
    return str(file_path)


class TestKnowledgeGraphDataset:
    def test_successful_load(self, mock_parquet_file, sample_df):
        # Arrange
        dataset = KnowledgeGraphDataset(filepath=mock_parquet_file)

        # Act
        result = dataset._load()

        # Assert
        assert isinstance(result, KnowledgeGraph)
        assert len(result._nodes) == len(sample_df)
        assert result._drug_nodes == ["drug1"]
        assert result._disease_nodes == ["disease1"]

    def test_file_not_found_retry_success(self):
        # Arrange
        dataset = KnowledgeGraphDataset(filepath="nonexistent.parquet")

        # Mock super()._load() to fail twice then succeed
        mock_df = pd.DataFrame(
            {"id": ["test"], "is_drug": [True], "is_disease": [False], "topological_embedding": [[1.0]]}
        )

        with patch.object(
            KnowledgeGraphDataset, "_load", side_effect=[FileNotFoundError, FileNotFoundError, KnowledgeGraph(mock_df)]
        ) as mock_load:
            # Act
            result = dataset._load()

            # Assert
            assert isinstance(result, KnowledgeGraph)
            assert mock_load.call_count == 3

    def test_file_not_found_all_retries_fail(self):
        # Arrange
        dataset = KnowledgeGraphDataset(filepath="nonexistent.parquet")

        # Act & Assert
        with pytest.raises(DatasetError) as exc_info:
            dataset._load()

        assert "Unable to find the Parquet file" in str(exc_info.value)

    def test_init_parameters(self):
        # Arrange
        test_args = {
            "filepath": "test.parquet",
            "load_args": {"key": "value"},
            "save_args": {"format": "parquet"},
            "version": None,
            "credentials": {"token": "123"},
            "fs_args": {"protocol": "s3"},
            "metadata": {"description": "test"},
        }

        # Act
        dataset = KnowledgeGraphDataset(**test_args)

        # Assert
        assert dataset._filepath == "test.parquet"
        assert dataset._load_args == {"key": "value"}
        assert dataset._save_args == {"format": "parquet"}
        assert dataset._credentials == {"token": "123"}
        assert dataset._fs_args == {"protocol": "s3"}
        assert dataset._metadata == {"description": "test"}
