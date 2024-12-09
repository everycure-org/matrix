import pytest
from unittest.mock import patch, MagicMock
from pipelines.matrix.src.matrix.datasets.graph import PandasParquetDataset
from kedro.io.core import DatasetError


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
