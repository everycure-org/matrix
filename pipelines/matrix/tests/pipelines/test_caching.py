from pathlib import Path
from typing import Callable, Sequence
from unittest.mock import AsyncMock, patch

import pytest
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset
from matrix.datasets.gcp import LazySparkDataset, PartitionedAsyncParallelDataset, SparkWithSchemaDataset
from matrix.pipelines.batch.pipeline import (
    cache_miss_resolver_wrapper,
    create_node_embeddings_pipeline,
    derive_cache_misses,
    limit_cache_to_results_from_api,
    lookup_from_cache,
    resolve_cache_duplicates,
)
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
def sample_input_df(spark: SparkSession, input_df_schema, sample_primary_key) -> DataFrame:
    data = [
        {sample_primary_key: "A", "category": "g"},
        {sample_primary_key: "B", "category": "h"},
        {sample_primary_key: "D", "category": "j"},
        {sample_primary_key: "E", "category": "k"},
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
def sample_cache(spark: SparkSession, cache_schema, sample_api1, sample_api2) -> DataFrame:
    data = [
        {"key": "A", "value": [1.0, 2.0], "api": sample_api1},
        {"key": "B", "value": [4.0, 5.0], "api": sample_api1},
        {"key": "C", "value": [7.0, 8.0], "api": sample_api2},
        {"key": "D", "value": [8.0, 9.0], "api": sample_api2},
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
def sample_cache_out(spark: SparkSession, cache_schema, sample_api1) -> DataFrame:
    data = [
        {"key": "E", "value": [5.0, 3.0], "api": sample_api1},
        {"key": "D", "value": [5.0, 3.0], "api": sample_api1},
    ]
    return spark.createDataFrame(data, schema=cache_schema)


@pytest.fixture
def sample_duplicate_cache(spark: SparkSession, cache_schema, sample_api1) -> DataFrame:
    data = [
        {"key": "A", "value": [1.0, 2.0], "api": sample_api1},
        {"key": "B", "value": [4.0, 5.0], "api": sample_api1},
        {"key": "B", "value": [4.0, 5.0], "api": sample_api1},
        {"key": "D", "value": [8.0, 9.0], "api": sample_api1},
        {"key": "E", "value": [9.0, 10.0], "api": sample_api1},
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
def sample_new_col():
    return "resolved"


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


@patch("matrix.pipelines.embeddings.encoders.DummyResolver.__new__", return_value=AsyncMock())
def test_cached_api_enrichment_pipeline(
    mock_encoder,
    sample_input_df: DataFrame,
    cache_schema: StructType,
    sample_api1: str,
    sample_primary_key: str,
    sample_preprocessor: Callable,
    sample_new_col: str,
    tmp_path: Path,
    kedro_session: KedroSession,
):
    """Verify the workings of the cached_api_enrichment_pipeline by calling it twice on the same dataset."""

    # Given a KedroSession, with a catalog where the cache is empty, and a
    # dataset where some keys need to be looked up using a service...
    async def dummy_resolver(docs: Sequence):
        return [[1.0, 2.0]] * len(docs)

    resolver = mock_encoder()  # Because of the patch, we are guaranteed that this object is _identical_ to the one that will be created by Kedro (through the custom inject decorator).
    resolver.apply.side_effect = dummy_resolver

    cache_path = str(tmp_path / "cache_dataset")
    catalog = DataCatalog(
        {
            "integration.prm.filtered_nodes": MemoryDataset(sample_input_df),
            "cache.read": SparkWithSchemaDataset(
                filepath=str(tmp_path / "cache_dataset"),
                provide_empty_if_not_present=True,
                load_args={"schema": cache_schema},
            ),
            "fully_enriched": LazySparkDataset(filepath=str(tmp_path / "enriched"), save_args={"mode": "overwrite"}),
            "cache_misses": LazySparkDataset(filepath=str(tmp_path / "cache_misses"), save_args={"mode": "overwrite"}),
            "cache.write": PartitionedAsyncParallelDataset(
                path=cache_path,
                dataset=ParquetDataset,
                filename_suffix=".parquet",
            ),
            "cache.reload": SparkWithSchemaDataset(
                filepath=cache_path,
                load_args={"schema": cache_schema},
            ),
            "params:caching.api": MemoryDataset(sample_api1),
            "params:caching.primary_key": MemoryDataset(sample_primary_key),
            "params:caching.preprocessor": MemoryDataset(sample_preprocessor),
            "params:caching.resolver": MemoryDataset({"_object": "matrix.pipelines.embeddings.encoders.DummyResolver"}),
            "params:caching.new_col": MemoryDataset(sample_new_col),
            "params:caching.batch_size": MemoryDataset(2),
        }
    )
    pipeline_run = create_node_embeddings_pipeline()

    runner = SequentialRunner()
    # ...when running the Kedro pipeline a first time...
    runner.run(pipeline_run, catalog)

    # ...then the data is found to be enriched...
    enriched_data = catalog.load("fully_enriched").toPandas()
    assert enriched_data[sample_new_col].isna().sum() == 0, "The input DataFrame should be fully enriched"
    assert enriched_data.shape == (4, 3)
    # ...by having the lookup service being called...
    resolver.apply.assert_called()
    # ...and because it's async, also awaited.
    resolver.apply.assert_awaited()
    # ... in other words, the catalog is complete.
    assert catalog.load("cache.read").toPandas().shape == (4, 3)  # Might need to force reloading the dataset.

    # When the pipeline is run a 2nd time...
    calls = resolver.apply.call_count
    awaits = resolver.apply.await_count
    runner.run(pipeline_run, catalog)
    # ... then the cache resolver should not have called out to the async function, as there is nothing to resolve.
    assert resolver.apply.call_count == calls
    assert resolver.apply.await_count == awaits


def test_no_resolver_calls_on_empty_cache_miss(spark: SparkSession):
    """Given that no keys resulted in a cache miss,
    when the cache_miss_resolver_wrapper is triggered automatically as a component in the cached_api_enrichment_pipeline,
    then no calls to the external API, through the cache miss resolver, will be made."""

    result = cache_miss_resolver_wrapper(
        df=spark.createDataFrame([], schema="key string"),
        transformer=AsyncMock(),
        api="foo",
        batch_size=1,
    )

    # That the encoder was not called, cannot be tested from here
    # as the cache_miss_resolver_wrapper only defers the actual lookups until
    # Kedro's partitioned dataset mechanism kicks in. That is done by the
    # AsyncParallelDataset's `save` call, which would only be tested from an
    # integration test, such as the one in `test_cached_api_enrichment_pipeline`.
    assert result == {}
