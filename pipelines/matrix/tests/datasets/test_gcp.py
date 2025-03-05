import pytest
from kedro.framework.session import KedroSession
from matrix.datasets.gcp import RemoteSparkJDBCDataset, SparkDatasetWithBQExternalTable, SparkWithSchemaDataset
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType


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


def test_spark_with_schema_dataset_bootstrap(kedro_session: KedroSession):
    """A SparkWithSchemaDataset can be bootstrapped if provide_empty_if_not_present is True"""
    structfield = "pyspark.sql.types.StructField"
    test_schema = {
        "schema": {
            "_object": "pyspark.sql.types.StructType",
            "fields": [
                {
                    "_object": structfield,
                    "name": "key",
                    "dataType": {"_object": "pyspark.sql.types.StringType"},
                    "nullable": False,
                },
                {
                    "_object": structfield,
                    "name": "value",
                    "dataType": {
                        "_object": "pyspark.sql.types.ArrayType",
                        "elementType": {
                            "_object": "pyspark.sql.types.FloatType",
                        },
                    },
                    "nullable": False,
                },
            ],
        }
    }
    dataset = SparkWithSchemaDataset(
        filepath="file:///tmp/non_existent_path",
        file_format="parquet",
        load_args=test_schema,
        provide_empty_if_not_present=True,
    )
    result_df = dataset.load()
    assert result_df.isEmpty()
    assert result_df.schema == StructType().add("key", StringType(), False).add("value", ArrayType(FloatType()), False)
