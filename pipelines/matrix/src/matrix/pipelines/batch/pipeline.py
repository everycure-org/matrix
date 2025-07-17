import hashlib
import itertools
import json
import logging
import warnings
from functools import partial
from typing import Any, Callable, Collection, Coroutine, Iterable, Iterator, Protocol, Sequence, TypeVar

import pyarrow as pa
from kedro.pipeline import Pipeline, pipeline
from matrix.inject import inject_object
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from matrix.pipelines.batch.schemas import to_spark_schema
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, count_distinct

T = TypeVar("T")
V = TypeVar("V")

logger = logging.getLogger(__name__)


class Transformer(Protocol):
    async def apply(self, strings: Collection[str], **kwargs) -> list: ...


def cached_api_enrichment_pipeline(
    source: str,
    workers: str | int,
    input: str,
    primary_key: str,
    cache_miss_resolver: str,  # Ref to [AttributeEncoder|Normalizer]
    preprocessor: str,  # Ref to Callable[[DataFrame], DataFrame],
    output: str,
    new_col: str,
    batch_size: str,
    cache_schema: str,
) -> Pipeline:
    """Define a Kedro Pipeline to enrich a Spark Dataframe using optionally cached API calls.

    The advantage to using this is that any identical API calls that have
    been made in other runs will have been cached, so that the enrichment
    process will complete faster.

    Note:
         ideally the preprocessor is a no-op, a simple pass-through (like
         `lambda x: x`). If you can shape your DataFrame in such a way that it
         doesn't need a custom preprocessor (e.g. by calling your own
         preprocessor in the node just prior to this pipeline), it'll be
         computationally more efficient (meaning this subpipeline will complete
         faster, and thus saves the organization money).

    Args:
        source: Any name that captures the context of the task, e.g. "embeddings".
            Note that this will be used to signal which cache to use, so if e.g.
            two embeddings steps require cached lookups but each lookup has a
            different return type from the resolver, then you should use two
            different source names.

        workers: The number of asynchronous tasks you want to use to split up
            the work given to the resolver.

        input: Kedro reference to a Spark DataFrame where you want to add a column
            `new_col` to.

        primary_key: Name of the column that should be produced (or passed) by the
            preprocessor function. It is this column of values that will be used to
            check against the cache, or failing that, sent to the `cache_miss_resolver`.

        cache_miss_resolver: Kedro reference to an object having an apply method,
            which is an asynchronous callable that will be used to look up any cache misses.

        api: Kedro parameter to restrict the cache to use results from this
            particular API. You will want to match this with the parameters of
            the `cache_miss_resolver`.

        preprocessor: Kedro reference to a callable that will preprocess the
            `input` such that it has a column `primary_key` which is used in the
            look-up process.

        output: Kedro reference to a dataset that is the input plus this new
            `new_col` containing the results from the enrichment with the cache/API.

        new_col: Name of the column in which the values associated with the
            `primary_key` should appear.

        batch_size: The size of a batch that will be sent to the embedder. Keep in
            mind that concurrent requests may be running, which means the API might be
            getting more batches in parallel, which in turn can  have an affect
            (positive or negative, it depends on the API) on the performance. This
            argument is a determining factor of the size of the files produced by running
            the cache miss resolver.

        cache_schema: A parameters entry referencing a PyArrow schema that is
            associated with the cache for this particular cached-lookup. Used for
            serialization purposes.

    Returns:
        Kedro Pipeline.
    """
    cache = f"batch.{source}.cache.read"
    cache_out = f"batch.{source}.{workers}.cache.write"
    cache_misses = f"batch.{source}.cache.misses"
    cache_reload = f"batch.{source}.cache.reload"

    nodes = [
        ArgoNode(
            name=f"derive_{source}_cache_misses",
            func=derive_cache_misses,
            inputs={
                "df": input,
                "cache": cache,
                "transformer": cache_miss_resolver,
                "primary_key": primary_key,
                "preprocessor": preprocessor,
                "cache_schema": cache_schema,
            },
            outputs=cache_misses,
            argo_config=ArgoResourceConfig(
                cpu_request=4,
                cpu_limit=4,
                memory_limit=16,
                memory_request=8,
            ),
        ),
        ArgoNode(
            name=f"resolve_{source}_cache_misses",
            func=cache_miss_resolver_wrapper,
            inputs={
                "df": cache_misses,
                "transformer": cache_miss_resolver,
                "batch_size": batch_size,
                "cache_schema": cache_schema,
            },
            outputs=cache_out,
            argo_config=ArgoResourceConfig(
                cpu_request=1,
                cpu_limit=1,
                memory_request=64,
                memory_limit=64,
            ),
        ),
        ArgoNode(
            name=f"lookup_{source}_from_cache",
            func=lookup_from_cache,
            inputs={
                "df": input,
                "cache": cache_reload,
                "transformer": cache_miss_resolver,
                "primary_key": primary_key,
                "preprocessor": preprocessor,
                "new_col": new_col,
                "lineage_dummy": cache_out,
            },
            outputs=output,
            tags=["argowf.fuse", f"argowf.fuse-group.{source}"],
            argo_config=ArgoResourceConfig(
                ephemeral_storage_limit=1024, ephemeral_storage_request=1024, memory_request=64, memory_limit=128
            ),
        ),
    ]

    return pipeline(nodes)


@inject_object()
def derive_cache_misses(
    df: DataFrame,
    cache: DataFrame,
    transformer: Transformer,
    primary_key: str,
    cache_schema: pa.lib.Schema,
    preprocessor: Callable[[DataFrame], DataFrame],
) -> DataFrame:
    api = transformer.version()
    if cache.isEmpty():
        # Replace the Kedro-loaded empty dataframe with meaningless schema by smt useful.
        cache = cache.sparkSession.createDataFrame([], to_spark_schema(cache_schema))
    report_on_cache_misses(cache, api)
    cache = limit_cache_to_results_from_api(cache, api=api).select("key")
    return (
        df.transform(preprocessor)
        .select(col(primary_key).alias("key"))
        .join(cache, on="key", how="leftanti")
        .distinct()
    )


@inject_object()
def cache_miss_resolver_wrapper(
    df: DataFrame, transformer: Transformer, batch_size: int, cache_schema: pa.lib.Schema
) -> dict[str, Callable[[], Coroutine[Any, Any, pa.Table]]]:
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps({"number of cache misses": df.count()}))
    documents = (_[0] for _ in df.select("key").toLocalIterator(prefetchPartitions=True))
    batches = batched(documents, batch_size)

    async def async_delegator(batch: Sequence[str]) -> pa.Table:
        logger.info(f"Processing batch with key: {batch[0]}")
        transformed = await transformer.apply(batch)
        logger.info(f"received response for batch with key: {batch[0]}")

        # Drop the api-field, since we're manually creating Hive partitions with that column.
        return pa.table(
            [batch, transformed],
            schema=cache_schema.remove(cache_schema.get_field_index("api")),
        ).to_pandas()

    def prep(
        batches: Iterable[Sequence[T]], api: str
    ) -> Iterator[tuple[str, Callable[[], Coroutine[Any, Any, pa.Table]]]]:
        for index, batch in enumerate(batches):
            head = batch[0]
            logger.info(f"materialized batch {index:>8_} of size {len(batch):_} with {head=}")
            # Create a unique filename per partition, with respect to the API
            key = hashlib.sha256((head + api).encode("utf-8")).hexdigest()
            yield key, partial(async_delegator, batch=batch)

    api = transformer.version()
    return {f"api={api}/{k}": v for k, v in prep(batches, api)}


@inject_object()
def lookup_from_cache(
    df: DataFrame,
    cache: DataFrame,
    transformer: Transformer,
    primary_key: str,
    preprocessor: Callable[[DataFrame], DataFrame],
    new_col: str,
    lineage_dummy: Any,  # required for kedro to keep the lineage: this function should come _after_ cache_miss_resolver_wrapper.
) -> DataFrame:
    api = transformer.version()
    report_on_cache_misses(cache, api)
    cache = (
        limit_cache_to_results_from_api(cache, api=api)
        .transform(resolve_cache_duplicates, id_col="key")
        .withColumnsRenamed({"key": primary_key, "value": new_col})
    )
    return df.transform(preprocessor).join(cache, how="left", on=primary_key)


def limit_cache_to_results_from_api(df: DataFrame, api: str) -> DataFrame:
    """Filter the cache for results from a particular `api`.

    Note that the reason for filtering for results from the api is that
    you likely don't want to add results from a different API than the one
    you use to resolve cache misses.
    """
    return df.filter(df["api"] == api).drop("api")


def resolve_cache_duplicates(df: DataFrame, id_col: str) -> DataFrame:
    """Return a DataFrame where duplicates have been removed in a non-discriminatory fashion.

    Duplicates are detected by their `id_col`.
    """
    keys, distinct_keys = df.agg(count("*"), count_distinct(id_col)).first()
    if keys != distinct_keys:
        # Warnings can be converted to errors, making them more ideal here.
        warnings.warn(
            "The cache contains duplicate keys. This is likely the result of a "
            "concurrent run and should be prevented. Continuing by non-"
            "discriminatory dropping duplicates…"
        )
        df = df.drop_duplicates([id_col])
    return df


def batched(iterable: Iterable[T], n: int, *, strict: bool = False) -> Iterator[tuple[T]]:
    # Taken from the recipe at https://docs.python.org/3/library/itertools.html#itertools.batched ,
    # which is available by default in Python 3.12
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("batch size must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def report_on_cache_misses(df: DataFrame, api: str) -> None:
    if logger.isEnabledFor(logging.INFO):
        rows = sorted(df.filter(df["api"] == api).groupBy(df["value"].isNull()).count().collect())
        if len(rows) > 1:
            no_nulls = rows[0][1]  # False sorts before True, so isNull->False comes first
            nulls = rows[1][1]
        elif len(rows) == 1:
            label, value = rows.pop()
            if label is False:
                no_nulls = value
                nulls = 0
            else:
                no_nulls = 0
                nulls = value
        else:
            nulls = 0
            no_nulls = 0
        logger.info(
            json.dumps({"api": api, "cache size": f"{no_nulls+nulls:_}", "non null cache values": f"{no_nulls:_}"})
        )


def pass_through(x):
    return x
