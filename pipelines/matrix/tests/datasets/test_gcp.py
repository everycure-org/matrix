import pytest
from matrix.datasets.gcp import RemoteSparkJDBCDataset, SparkDatasetWithBQExternalTable


def test_remote_spark_dataset_split_path():
    # Given an instance of the RemoteSparkJDBCDataset
    url = "sqlite:gs://bucket/path/to/file.csv"

    # When splitting the jdbc path
    protocol, fs_prefix, blob_name = RemoteSparkJDBCDataset.split_remote_jdbc_path(url)

    # Then correct pieces extracted
    assert protocol == "sqlite"
    assert fs_prefix == "gs://"
    assert blob_name == "bucket/path/to/file.csv"


@pytest.mark.parametrize(
    "identifier,expected",
    [
        # Correctly replace dash
        ("bq-table", "bq_table"),
        # Correctly replace dot
        ("bq.table", "bq_table"),
        # Correctly replace both
        ("gcp-bq.table", "gcp_bq_table"),
    ],
)
def test_sanitize_bq_strings(identifier, expected):
    assert SparkDatasetWithBQExternalTable._sanitize_name(identifier) == expected
