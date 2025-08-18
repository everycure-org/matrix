import kedro.io.core
import pytest
from kedro.framework.session import KedroSession
from matrix_gcp_datasets.gcp import LazySparkDataset, RemoteSparkJDBCDataset, SparkDatasetWithBQExternalTable
from pyspark.sql.types import BooleanType, StructType


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


def test_lazysparkdataset_creation_when_missing(kedro_session: KedroSession):
    """A dummy DataFrame is provided for a LazySparkDataset if provide_empty_if_not_present is True"""
    dataset = LazySparkDataset(
        filepath="file:///tmp/non_existent_path",
        file_format="parquet",
        provide_empty_if_not_present=True,
    )
    result_df = dataset.load()
    assert result_df.isEmpty()
    assert result_df.schema == StructType().add("foo", BooleanType(), True)


def test_lazysparkdataset_error_when_missing(kedro_session: KedroSession):
    """The usual Kedro exception is raised on loading a LazySparkDataset, when provide_empty_if_not_present is False"""
    dataset = LazySparkDataset(
        filepath="file:///tmp/non_existent_path",
        file_format="parquet",
        provide_empty_if_not_present=False,
    )
    with pytest.raises(kedro.io.core.DatasetError):
        dataset.load()
