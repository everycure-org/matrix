import os
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from kedro_datasets.spark import SparkDataset
from matrix.datasets.gcp import LazySparkDataset, SparkWithSchemaDataset
from matrix.pipelines.batch.pipeline import (
    cache_miss_resolver_wrapper,
    cached_api_enrichment_pipeline,
    derive_cache_misses,
    dummy_resolver,
    limit_cache_to_results_from_api,
    lookup_from_cache,
    pass_through,
    resolve_cache_duplicates,
)
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
        {
            "key": "B",
            "value": [
                4.0,
                5.0,
            ],
            "api": "gpt-4",
        },
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
        {
            "key": "B",
            "value": [
                4.0,
                5.0,
            ],
            "api": "gpt-4",
        },
        {"key": "D", "value": [8.0, 9.0], "api": "gpt-3"},
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
def sample_id_col():
    return "key"


@pytest.fixture
def sample_preprocessor():
    return pass_through


@pytest.fixture
def sample_resolver():
    return dummy_resolver


@pytest.fixture
def sample_new_col():
    return "resolved"


@pytest.fixture
def mock_encoder(sample_api1) -> Mock:
    encoder = Mock(side_effect=partial(dummy_resolver, api=sample_api1))
    return encoder


@pytest.fixture
def mock_encoder2(sample_api1) -> Mock:
    encoder = Mock(side_effect=partial(dummy_resolver, api=sample_api1))
    return encoder


def test_spark_with_schema_dataset_bootstrap(test_schema, cache_schema):
    # Define the path
    project_path = Path.cwd()

    # Bootstrap the project to set up the configuration
    bootstrap_project(project_path)

    # Configure the project
    configure_project("matrix")

    # Create a Kedro session and context
    with KedroSession.create(project_path) as session:
        kedro_context = session.load_context()

        # Define a non-existent filepath to trigger the bootstrap behavior
        non_existent_path = "file:///tmp/non_existent_path"

        # Create an instance of SparkWithSchemaDataset with bootstrapping enabled
        dataset = SparkWithSchemaDataset(
            filepath=non_existent_path,
            file_format="parquet",
            load_args=test_schema,
            provide_empty_if_not_present=True,
        )

        # Attempt to load the dataset
        result_df = dataset.load()
        # Verify that the result is an empty DataFrame with the correct schema
        assert result_df.rdd.isEmpty(), "The DataFrame should be empty"
        assert result_df.schema == cache_schema, "The schema should match the provided schema"


def test_derive_cache_misses(
    sample_input_df, sample_cache, sample_api1, sample_primary_key, sample_preprocessor, sample_cache_misses
):
    result_df = derive_cache_misses(sample_input_df, sample_cache, sample_api1, sample_primary_key, sample_preprocessor)
    assertDataFrameEqual(result_df, sample_cache_misses)


def test_dataframe_is_enriched(
    sample_cache,
    sample_cache_out,
    sample_cache_misses,
    sample_resolver,
    sample_api1,
    sample_input_df,
    sample_primary_key,
    sample_preprocessor,
):
    sample_cache.show()
    sample_cache.printSchema()
    sample_cache_misses.show()
    cache_out = cache_miss_resolver_wrapper(sample_cache_misses, partial(sample_resolver, api=sample_api1), sample_api1)
    cache_out.printSchema()
    assert isinstance(cache_out, DataFrame)
    # cache_out.show()
    assertDataFrameEqual(cache_out, sample_cache_out)
    new_cache = sample_cache.union(cache_out)
    cache_misses2 = derive_cache_misses(
        sample_input_df, new_cache, sample_api1, sample_primary_key, sample_preprocessor
    )
    assert cache_misses2.count() == 0, "The cache misses should be empty"


def test_cache_contains_duplicates_warning(sample_duplicate_cache, sample_id_col):
    # Capture the warning
    with pytest.warns(UserWarning, match="The cache contains duplicate keys."):
        resolve_cache_duplicates(sample_duplicate_cache, sample_id_col)


def test_enriched_keeps_same_size_with_cache_duplicates(
    sample_input_df,
    sample_cache,
    sample_duplicate_cache,
    sample_api1,
    sample_primary_key,
    sample_preprocessor,
    sample_resolver,
    sample_new_col,
):
    cache_misses = derive_cache_misses(
        sample_input_df, sample_duplicate_cache, sample_api1, sample_primary_key, sample_preprocessor
    )
    cache_out = cache_miss_resolver_wrapper(cache_misses, partial(sample_resolver, api=sample_api1), sample_api1)
    new_cache = sample_cache.union(cache_out)
    enriched_df = lookup_from_cache(
        sample_input_df, new_cache, sample_api1, sample_primary_key, sample_preprocessor, sample_new_col
    )
    assert (
        enriched_df.count() == sample_input_df.count()
    ), "The enriched DataFrame should have the same number of rows as the input DataFrame"


def test_different_api(sample_cache, sample_api1, api1_filtered_cache, sample_api2, api2_filtered_cache):
    result1_df = limit_cache_to_results_from_api(sample_cache, sample_api1)
    assertDataFrameEqual(result1_df, api1_filtered_cache)
    result2_df = limit_cache_to_results_from_api(sample_cache, sample_api2)
    assertDataFrameEqual(result2_df, api2_filtered_cache)


def test_re_resolve_is_noop(
    sample_primary_key,
    sample_preprocessor,
    sample_cache,
    sample_input_df,
    mock_encoder,
    mock_encoder2,
    sample_cache_misses,
    sample_api1,
):
    cache_out = cache_miss_resolver_wrapper(sample_cache_misses, mock_encoder, sample_api1)
    new_cache = sample_cache.union(cache_out)
    cache_misses2 = derive_cache_misses(
        sample_input_df, new_cache, sample_api1, sample_primary_key, sample_preprocessor
    )
    result2 = cache_miss_resolver_wrapper(cache_misses2, mock_encoder2, sample_api1)
    mock_encoder2.assert_not_called()


# def test_cached_api_enrichment_pipeline(sample_input_df,
#                                         sample_primary_key,
#                                         sample_api1,
#                                         sample_new_col,
#                                         mock_encoder,
#                                         mock_encoder2,
#                                         sample_preprocessor,
#                                         test_schema):
#     project_path = Path.cwd()
#     bootstrap_project(project_path)
#     configure_project("matrix")
#     with KedroSession.create(project_path) as session:
#         kedro_context = session.load_context()
#         temp_dir = tempfile.TemporaryDirectory().name

#         test_cache = SparkWithSchemaDataset(
#                     filepath=os.path.join(temp_dir, "cache_dataset"),
#                     provide_empty_if_not_present=True,
#                     load_args={"schema": test_schema},
#                     )

#         test_cache_write = SparkWithSchemaDataset(
#                             filepath=os.path.join(temp_dir, "cache_dataset"),
#                             provide_empty_if_not_present=True,
#                             load_args={"schema": test_schema},
#                             save_args={"mode": "append", "partitionBy": ["api"]}
#         )

#         test_cache_misses = LazySparkDataset(
#                             filepath=os.path.join(temp_dir, "cache_misses"),
#                             save_args={"mode": "overwrite"}
#                         )

#         test_fully_enriched = LazySparkDataset(
#                                 filepath=os.path.join(temp_dir, "fully_enriched"),
#                                 save_args={"mode": "overwrite"}
#                             )

#         pipeline = cached_api_enrichment_pipeline(input="caching.input",
#                                                 cache="cache.read",
#                                                 cache_out="cache.write",
#                                                 cache_misses="cache.misses",
#                                                 primary_key="params:caching.primary_key",
#                                                 cache_miss_resolver="params:caching.resolver",
#                                                 preprocessor="params:caching.preprocessor",
#                                                 api="params:caching.api",
#                                                 output="fully_enriched",
#                                                 new_col="params:caching.new_col")

#         mock_catalog1 = DataCatalog({
#             "caching.input": LazySparkDataset(filepath=os.path.join(temp_dir, "input_dataset")),
#             "cache.read": test_cache,
#             "cache.write": test_cache_write,
#             "cache_misses": test_cache_misses,
#             "params:caching.primary_key": sample_primary_key,
#             "params:caching.resolver": mock_encoder,
#             "params:caching.api": sample_api1,
#             "params:caching.preprocessor": sample_preprocessor,
#             "fully_enriched": test_fully_enriched,
#             "params:caching.new_col": sample_new_col,
#         })

#         mock_catalog1.save("caching.input", sample_input_df)

#         result1 = SequentialRunner().run(pipeline, mock_catalog1)

#         result1["cache.read"].load().show()

#         mock_catalog2 = DataCatalog({
#             "caching.input": LazySparkDataset(filepath=os.path.join(temp_dir, "input_dataset")),
#             "cache.read": test_cache,
#             "cache.write": test_cache_write,
#             "cache_misses": test_cache_misses,
#             "params:caching.primary_key": sample_primary_key,
#             "params:caching.resolver": mock_encoder2,
#             "params:caching.api": sample_api1,
#             "params:caching.preprocessor": sample_preprocessor,
#             "fully_enriched": test_fully_enriched,
#             "params:caching.new_col": sample_new_col,
#         })

#         mock_catalog2.save("caching.input", sample_input_df)

#         result2 = SequentialRunner().run(pipeline, mock_catalog2)
#         mock_encoder2.assert_not_called()
