import pytest

from matrix.datasets.gcp import RemoteSparkJDBCDataset


def test_remote_spark_dataset_split_path():
    # Given an instance of the RemoteSparkJDBCDataset
    url = "sqlite:gs://bucket/path/to/file.csv"

    # When splitting the jdbc path
    protocol, fs_prefix, blob_name = RemoteSparkJDBCDataset.split_remote_jdbc_path(url)

    # Then correct pieces extracted
    assert protocol == "sqlite"
    assert fs_prefix == "gs://"
    assert blob_name == "bucket/path/to/file.csv"
