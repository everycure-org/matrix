import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pyspark as ps
import pytest
from kedro.framework.context import KedroContext
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset
from matrix.datasets.gcp import LazySparkDataset, PartitionedAsyncParallelDataset, SparkWithSchemaDataset
from matrix.pipelines.batch.pipeline import (
    create_node_embeddings_pipeline,
    derive_cache_misses,
    limit_cache_to_results_from_api,
    lookup_from_cache,
    resolve_cache_duplicates,
)
from matrix.pipelines.embeddings.encoders import DummyResolver
from matrix.pipelines.embeddings.nodes import pass_through
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def test_schema():
    mock_schema = {
        "schema": {
            "_object": "pyspark.sql.types.StructType",
            "fields": [
                {
                    "_object": "pyspark.sql.types.StructField",
                    "name": "key",
                    "dataType": {
                        "_object": "pyspark.sql.types.StringType",
                    },
                    "nullable": False,
                },
                {
                    "_object": "pyspark.sql.types.StructField",
                    "name": "value",
                    "dataType": {
                        "_object": "pyspark.sql.types.ArrayType",
                        "elementType": {
                            "_object": "pyspark.sql.types.FloatType",
                        },
                    },
                    "nullable": False,
                },
                {
                    "_object": "pyspark.sql.types.StructField",
                    "name": "api",
                    "dataType": {
                        "_object": "pyspark.sql.types.StringType",
                    },
                    "nullable": False,
                },
            ],
        }
    }
    return mock_schema


@pytest.fixture
def input_df_schema():
    return StructType(
        [
            StructField("to_resolve", StringType(), False),
            StructField("category", StringType(), False),
        ]
    )


@pytest.fixture
def sample_input_df(spark: SparkSession, input_df_schema) -> DataFrame:
    data = [
        {"to_resolve": "A", "category": "g"},
        {"to_resolve": "B", "category": "h"},
        {"to_resolve": "D", "category": "j"},
        {"to_resolve": "E", "category": "k"},
    ]
    return spark.createDataFrame(data, input_df_schema)


@pytest.fixture
def sample_primary_key():
    return "to_resolve"


@pytest.fixture
def cache_schema():
    return StructType(
        [
            StructField("key", StringType(), False),
            StructField("value", ArrayType(FloatType()), False),
            StructField("api", StringType(), False),
        ]
    )


@pytest.fixture
def filtered_cache_schema():
    return StructType(
        [
            StructField("key", StringType(), False),
            StructField("value", ArrayType(FloatType()), False),
        ]
    )


@pytest.fixture
def cache_misses_schema():
    return StructType(
        [
            StructField("key", StringType(), False),
        ]
    )


@pytest.fixture
def sample_cache(spark: SparkSession, cache_schema) -> DataFrame:
    data = [
        {"key": "A", "value": [1.0, 2.0], "api": "gpt-4"},
        {"key": "B", "value": [4.0, 5.0], "api": "gpt-4"},
        {"key": "C", "value": [7.0, 8.0], "api": "gpt-3"},
        {"key": "D", "value": [8.0, 9.0], "api": "gpt-3"},
    ]
    return spark.createDataFrame(data, schema=cache_schema)


@pytest.fixture
def api1_filtered_cache(spark: SparkSession, filtered_cache_schema) -> DataFrame:
    data = [
        {"key": "A", "value": [1.0, 2.0]},
        {
            "key": "B",
            "value": [
                4.0,
                5.0,
            ],
        },
    ]
    return spark.createDataFrame(data, schema=filtered_cache_schema)


@pytest.fixture
def api2_filtered_cache(spark: SparkSession, filtered_cache_schema) -> DataFrame:
    data = [
        {"key": "C", "value": [7.0, 8.0]},
        {"key": "D", "value": [8.0, 9.0]},
    ]
    return spark.createDataFrame(data, schema=filtered_cache_schema)


@pytest.fixture
def sample_cache_out(spark: SparkSession, cache_schema) -> DataFrame:
    data = [
        {"key": "E", "value": [5.0, 3.0], "api": "gpt-4"},
        {"key": "D", "value": [5.0, 3.0], "api": "gpt-4"},
    ]
    return spark.createDataFrame(data, schema=cache_schema)


@pytest.fixture
def sample_duplicate_cache(spark: SparkSession, cache_schema) -> DataFrame:
    data = [
        {"key": "A", "value": [1.0, 2.0], "api": "gpt-4"},
        {"key": "B", "value": [4.0, 5.0], "api": "gpt-4"},
        {"key": "B", "value": [4.0, 5.0], "api": "gpt-4"},
        {"key": "D", "value": [8.0, 9.0], "api": "gpt-4"},
        {"key": "E", "value": [9.0, 10.0], "api": "gpt-4"},
    ]
    return spark.createDataFrame(data, schema=cache_schema)


@pytest.fixture
def sample_cache_misses(spark: SparkSession, cache_misses_schema) -> DataFrame:
    data = [
        {"key": "D"},
        {"key": "E"},
    ]
    return spark.createDataFrame(data, schema=cache_misses_schema)


@pytest.fixture
def sample_api1():
    return "gpt-4"


@pytest.fixture
def sample_api2():
    return "gpt-3"


@pytest.fixture
def sample_batch_size():
    return 2


@pytest.fixture
def sample_id_col():
    return "key"


@pytest.fixture
def sample_preprocessor():
    return pass_through


@pytest.fixture
def sample_resolver():
    return DummyResolver


@pytest.fixture
def sample_new_col():
    return "resolved"


@pytest.fixture
def mock_encoder() -> Mock:
    encoder = Mock(side_effect=DummyResolver)
    return encoder


@pytest.fixture
def mock_encoder2() -> Mock:
    encoder = Mock(side_effect=DummyResolver)
    return encoder


def test_spark_with_schema_dataset_bootstrap(test_schema, cache_schema):
    project_path = Path.cwd()

    # Bootstrap the project to set up the configuration
    bootstrap_project(project_path)
    configure_project("matrix")

    # Create a Kedro session and context
    with KedroSession.create(project_path) as session:
        kedro_context = session.load_context()
        non_existent_path = "file:///tmp/non_existent_path"
        dataset = SparkWithSchemaDataset(
            filepath=non_existent_path,
            file_format="parquet",
            load_args=test_schema,
            provide_empty_if_not_present=True,
        )
        result_df = dataset.load()
        assert result_df.rdd.isEmpty(), "The DataFrame should be empty"
        assert result_df.schema == cache_schema, "The schema should match the provided schema"


def test_derive_cache_misses(
    sample_input_df, sample_cache, sample_api1, sample_primary_key, sample_preprocessor, sample_cache_misses
):
    result_df = derive_cache_misses(sample_input_df, sample_cache, sample_api1, sample_primary_key, sample_preprocessor)
    assertDataFrameEqual(result_df, sample_cache_misses)


def test_cache_contains_duplicates_warning(sample_duplicate_cache, sample_id_col):
    # Capture the warning
    with pytest.warns(UserWarning, match="The cache contains duplicate keys."):
        resolve_cache_duplicates(sample_duplicate_cache, sample_id_col)


def test_enriched_keeps_same_size_with_cache_duplicates(
    sample_input_df,
    sample_duplicate_cache,
    sample_api1,
    sample_primary_key,
    sample_preprocessor,
    sample_new_col,
):
    enriched_df = lookup_from_cache(
        sample_input_df,
        sample_duplicate_cache,
        sample_api1,
        sample_primary_key,
        sample_preprocessor,
        sample_new_col,
        lineage_dummy="foo",
    )
    assert (
        enriched_df.count() == sample_input_df.count()
    ), "The enriched DataFrame should have the same number of rows as the input DataFrame"


def test_different_api(sample_cache, sample_api1, api1_filtered_cache, sample_api2, api2_filtered_cache):
    result1_df = limit_cache_to_results_from_api(sample_cache, sample_api1)
    assertDataFrameEqual(result1_df, api1_filtered_cache)
    result2_df = limit_cache_to_results_from_api(sample_cache, sample_api2)
    assertDataFrameEqual(result2_df, api2_filtered_cache)


# @pytest.mark.parametrize("kedro_context", ["cloud_kedro_context", "base_kedro_context"])
def test_df_fully_enriched(
    sample_input_df,
    cache_schema,
    sample_api1,
    sample_primary_key,
    sample_preprocessor,
    sample_batch_size,
    sample_new_col,
    mock_encoder,
    mock_encoder2,
    spark,
    # kedro_context: KedroContext,
    # request: pytest.FixtureRequest,
):
    # if 'spark' in globals():
    #     spark.stop()

    # spark = (
    #     SparkSession.builder
    #     .master("local[*]")
    #     .config("spark.driver.host", "127.0.0.1")
    #     .config("spark.driver.bindAddress", "127.0.0.1")
    #     .config("spark.ui.enabled", "false")
    #     .config("spark.local.dir", "/tmp/spark-temp")
    #     .config("spark.network.timeout", "10000s")
    #     .config("spark.sql.shuffle.partitions", "1")
    #     .config("spark.driver.port", "4040")
    #     .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv6Addresses=false")
    #     .getOrCreate()
    # )

    project_path = Path.cwd()
    bootstrap_project(project_path)
    configure_project("matrix")
    # kedro_context = request.getfixturevalue(kedro_context)
    with KedroSession.create(project_path) as session:
        kedro_context = session.load_context()
        temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory: {temp_dir}")
        print(spark.sparkContext.getConf().getAll())
        input = {
            "integration.prm.filtered_nodes": MemoryDataset(sample_input_df),
            "cache.read": SparkWithSchemaDataset(
                filepath=os.path.join(temp_dir, "cache_dataset"),
                provide_empty_if_not_present=True,
                load_args={"schema": cache_schema},
            ),
            "test_fully_enriched": LazySparkDataset(
                filepath=os.path.join(temp_dir, "fully_enriched"), save_args={"mode": "overwrite"}
            ),
            "cache_misses": LazySparkDataset(
                filepath=os.path.join(temp_dir, "cache_misses"), save_args={"mode": "overwrite"}
            ),
            "cache.write": PartitionedAsyncParallelDataset(
                path=os.path.join(temp_dir, "cache_dataset"),
                dataset=ParquetDataset,
                filename_suffix=".parquet",
            ),
            "cache.reload": SparkWithSchemaDataset(
                filepath=os.path.join(temp_dir, "cache_dataset"),
                load_args={"schema": cache_schema},
            ),
            "params:caching.api": sample_api1,
            "params:caching.primary_key": sample_primary_key,
            "params:caching.preprocessor": sample_preprocessor,
            "params:caching.resolver": mock_encoder,
            "params:caching.new_col": sample_new_col,
            "params:caching.batch_size": sample_batch_size,
        }
        mock_catalog_run1 = DataCatalog(input)
        # print(mock_catalog_run1.list())
        pipeline_run1 = create_node_embeddings_pipeline()
        runner = SequentialRunner()
        runner.run(pipeline_run1, mock_catalog_run1)
        enriched_data = mock_catalog_run1.load("fully_enriched").toPandas()
        assert enriched_data[sample_new_col].isNull().sum() == 0, "The input DataFrame should be fully enriched"
        # Run the pipeline the 2nd time
        mock_catalog_run2 = DataCatalog(input | {"params:caching.resolver": mock_encoder2})
        pipeline_run2 = create_node_embeddings_pipeline()
        runner.run(pipeline_run2, mock_catalog_run2)
        mock_encoder.assert_called()  # The first run, encoder should be called
        mock_encoder2.assert_not_called()  # The second run, encoder should not be called
