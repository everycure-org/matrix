from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pyarrow as pa
import pytest
from kedro.framework.session import KedroSession
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import ParquetDataset
from matrix.datasets.gcp import LazySparkDataset, PartitionedAsyncParallelDataset
from matrix.pipelines.batch.pipeline import (
    cache_miss_resolver_wrapper,
    derive_cache_misses,
    limit_cache_to_results_from_api,
    lookup_from_cache,
    pass_through,
    resolve_cache_duplicates,
)
from matrix.pipelines.embeddings.encoders import DummyResolver
from matrix.pipelines.embeddings.pipeline import create_node_embeddings_pipeline
from matrix.pipelines.integration.normalizers.normalizers import DummyNodeNormalizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual


@pytest.fixture
def embeddings_schema() -> pa.lib.Schema:
    return pa.schema({"key": pa.string(), "value": pa.list_(pa.float32()), "api": pa.string()})


@pytest.fixture
def sample_input_df(spark: SparkSession, sample_primary_key) -> DataFrame:
    schema = StructType(
        [
            StructField(sample_primary_key, StringType(), False),
            StructField("category", StringType(), False),
        ]
    )
    data = [("A", "g"), ("B", "h"), ("D", "j"), ("E", "k")]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_primary_key():
    return "to_resolve"


@pytest.fixture
def cache_schema(filtered_cache_schema):
    return deepcopy(filtered_cache_schema).add("api", StringType(), False)


@pytest.fixture
def filtered_cache_schema():
    return StructType().add("key", StringType(), False).add("value", ArrayType(FloatType()), False)


@pytest.fixture
def sample_cache(spark: SparkSession, cache_schema, sample_api1, sample_api2) -> DataFrame:
    data = [
        ("A", [1.0, 2.0], sample_api1.version()),
        ("B", [4.0, 5.0], sample_api1.version()),
        ("C", [7.0, 8.0], sample_api2),
        ("D", [8.0, 9.0], sample_api2),
    ]
    return spark.createDataFrame(data, schema=cache_schema)


@pytest.fixture
def sample_duplicate_cache(spark: SparkSession, cache_schema, sample_api1) -> DataFrame:
    data = [
        ("A", [1.0, 2.0], sample_api1.version()),
        ("B", [4.0, 5.0], sample_api1.version()),
        ("B", [4.0, 5.0], sample_api1.version()),
        ("D", [8.0, 9.0], sample_api1.version()),
        ("E", [9.0, 10.0], sample_api1.version()),
    ]
    return spark.createDataFrame(data, schema=cache_schema)


@pytest.fixture
def sample_api1():
    return DummyNodeNormalizer(True, True)


@pytest.fixture
def sample_api2():
    return "gpt-3"


@pytest.fixture
def sample_api3():
    return DummyResolver()


@pytest.fixture
def sample_id_col():
    return "key"


@pytest.fixture
def sample_new_col():
    return "resolved"


def test_derive_cache_misses(sample_input_df, sample_cache, sample_api1, sample_primary_key, embeddings_schema, spark):
    expected = spark.createDataFrame(
        [
            ("D",),
            ("E",),
        ],
        schema=("key",),
    )

    result_df = derive_cache_misses(
        df=sample_input_df,
        cache=sample_cache,
        transformer=sample_api1,
        primary_key=sample_primary_key,
        preprocessor=pass_through,
        cache_schema=embeddings_schema,
    )
    assertDataFrameEqual(result_df, expected)


def test_cache_contains_duplicates_warning(sample_duplicate_cache, sample_id_col):
    # Capture the warning
    with pytest.warns(UserWarning, match="The cache contains duplicate keys."):
        resolve_cache_duplicates(sample_duplicate_cache, sample_id_col)


def test_enriched_keeps_same_size_with_cache_duplicates(
    sample_input_df,
    sample_duplicate_cache,
    sample_api1,
    sample_primary_key,
    sample_new_col,
):
    enriched_df = lookup_from_cache(
        sample_input_df,
        sample_duplicate_cache,
        sample_api1.version(),
        sample_primary_key,
        pass_through,
        sample_new_col,
        lineage_dummy="foo",
    )
    assert (
        enriched_df.count() == sample_input_df.count()
    ), "The enriched DataFrame should have the same number of rows as the input DataFrame"


def test_different_api(sample_cache, sample_api2, spark: SparkSession, filtered_cache_schema: StructType):
    result1_df = limit_cache_to_results_from_api(sample_cache, sample_api2)
    data = [
        ("C", [7.0, 8.0]),
        ("D", [8.0, 9.0]),
    ]
    expected = spark.createDataFrame(data, schema=filtered_cache_schema)
    assertDataFrameEqual(result1_df, expected)


@pytest.mark.integration
@patch("matrix.pipelines.embeddings.encoders.DummyResolver.__new__", return_value=AsyncMock())
def test_cached_api_enrichment_pipeline(
    mock_encoder,
    sample_input_df: DataFrame,
    cache_schema: StructType,
    embeddings_schema: pa.lib.Schema,
    sample_primary_key: str,
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

    output = "embeddings.feat.graph.node_embeddings@spark"
    cache_path = str(tmp_path / "cache_dataset")
    catalog = DataCatalog(
        {
            "filtering.prm.filtered_nodes": MemoryDataset(sample_input_df),
            "batch.node_embeddings.cache.read": LazySparkDataset(
                filepath=str(tmp_path / "cache_dataset"),
                provide_empty_if_not_present=True,
                load_args={"schema": cache_schema},
            ),
            output: LazySparkDataset(filepath=str(tmp_path / "enriched"), save_args={"mode": "overwrite"}),
            "batch.node_embeddings.cache_misses": LazySparkDataset(
                filepath=str(tmp_path / "cache_misses"), save_args={"mode": "overwrite"}
            ),
            "batch.node_embeddings.20.cache.write": PartitionedAsyncParallelDataset(
                path=cache_path,
                dataset=ParquetDataset,
                filename_suffix=".parquet",
            ),
            "batch.node_embeddings.cache.reload": LazySparkDataset(
                filepath=cache_path,
            ),
            "params:embeddings.node.primary_key": MemoryDataset(sample_primary_key),
            "params:embeddings.node.preprocessor": MemoryDataset(pass_through),
            "params:embeddings.node.resolver": MemoryDataset(
                {"_object": "matrix.pipelines.embeddings.encoders.DummyResolver"}
            ),
            "params:embeddings.node.target_col": MemoryDataset(sample_new_col),
            "params:embeddings.node.batch_size": MemoryDataset(2),
            "params:embeddings.node.cache_schema": MemoryDataset(embeddings_schema),
        }
    )
    pipeline_run = create_node_embeddings_pipeline()

    runner = SequentialRunner()
    # ...when running the Kedro pipeline a first time...
    runner.run(pipeline_run, catalog)

    # ...then the data is found to be enriched...
    enriched_data = catalog.load(output).toPandas()
    assert enriched_data[sample_new_col].isna().sum() == 0, "The input DataFrame should be fully enriched"
    assert enriched_data.shape == (4, 3)
    # ...by having the lookup service being called...
    resolver.apply.assert_called()
    # ...and because it's async, also awaited.
    resolver.apply.assert_awaited()
    # ... in other words, the catalog is now fully seeded.
    assert catalog.load(output).toPandas().shape == (4, 3)

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
        batch_size=1,
        cache_schema=pa.schema({"foo": pa.string()}),
    )

    # That the encoder was not called, cannot be tested from here
    # as the cache_miss_resolver_wrapper only defers the actual lookups until
    # Kedro's partitioned dataset mechanism kicks in. That is done by the
    # AsyncParallelDataset's `save` call, which would only be tested from an
    # integration test, such as the one in `test_cached_api_enrichment_pipeline`.
    assert result == {}
