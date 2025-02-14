import logging
import warnings
from typing import Any, Callable, Collection, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar

import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from matrix.inject import inject_object
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.window import Window

T = TypeVar("T")
V = TypeVar("V")


@inject_object()
def _transform(dfs: Dict[str, Any], transformer, **transformer_kwargs):
    """Function to bucketize input data.

    Args:
        dfs: mapping of paths to df load functions
        encoder: encoder to run
    """

    def _func(dataframe: pd.DataFrame):
        return lambda df=dataframe: transformer.apply(df(), **transformer_kwargs)

    shards = {}
    for path, df in dfs.items():
        # Invoke function to compute embeddings
        shards[path] = _func(df)

    return shards


def _bucketize(df: DataFrame, bucket_size: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Function to bucketize df in given number of buckets.

    Args:
        df: dataframe to bucketize
        bucket_size: size of the buckets
    Returns:
        Dataframe augmented with `bucket` column
    """

    if columns is None:
        columns = []

    # Retrieve number of elements
    num_elements = df.count()
    num_buckets = (num_elements + bucket_size - 1) // bucket_size

    # Construct df to bucketize
    spark_session: SparkSession = SparkSession.builder.getOrCreate()

    # Bucketize df
    buckets = spark_session.createDataFrame(
        data=[(bucket, bucket * bucket_size, (bucket + 1) * bucket_size) for bucket in range(num_buckets)],
        schema=["bucket", "min_range", "max_range"],
    )

    return (
        df.withColumn("row_num", F.row_number().over(Window.orderBy("id")) - F.lit(1))
        .join(buckets, on=[(F.col("row_num") >= (F.col("min_range"))) & (F.col("row_num") < F.col("max_range"))])
        .select("id", *columns, "bucket")
    )


def create_pipeline(
    source: str,
    df: str,
    output: str,
    bucket_size: str,
    transformer: str,
    columns: List[str] = None,
    max_workers: int = 10,
    **transformer_kwargs,
) -> Pipeline:
    """Pipeline to transform dataframe."""
    return pipeline(
        [
            ArgoNode(
                func=_bucketize,
                inputs={
                    key: value
                    for key, value in {"df": df, "bucket_size": bucket_size, "columns": columns}.items()
                    if value is not None
                },
                outputs=f"batch.int.{source}.input_bucketized@spark",
                name=f"bucketize_{source}_input",
                argo_config=ArgoResourceConfig(
                    cpu_request=48,
                    cpu_limit=48,
                    memory_limit=192,
                    memory_request=120,
                ),
            ),
            node(
                func=_transform,
                inputs={
                    "dfs": f"batch.int.{source}.input_bucketized@partitioned",
                    "transformer": transformer,
                    **transformer_kwargs,
                },
                outputs=f"batch.int.{source}.{max_workers}.input_transformed@partitioned",
                name=f"transform_{source}_input",
            ),
            node(
                func=lambda x: x,
                inputs=[f"batch.int.{source}.{max_workers}.input_transformed@spark"],
                outputs=output,
                name=f"extract_{source}_input",
            ),
        ],
        tags=["argowf.fuse", f"argowf.fuse-group.{source}"],
    )


CACHE_COLUMNS = ["key", "value", "api"]


@inject_object()
def derive_cache_misses(
    df: DataFrame, cache: DataFrame, api: str, primary_key: str, preprocessor: Callable[[DataFrame], DataFrame]
) -> DataFrame:
    assert (
        cache.columns == CACHE_COLUMNS
    ), f"The cache's columns does not match {CACHE_COLUMNS}. Note that the order is fixed, so that appends would work correctly."
    cache = limit_cache_to_results_from_api(cache, api=api).select(CACHE_COLUMNS[0])
    return (
        df.transform(preprocessor)
        .select(F.col(primary_key).alias(CACHE_COLUMNS[0]))
        .join(cache, on=CACHE_COLUMNS[0], how="leftanti")
        .distinct()
    )


@inject_object()
def cache_miss_resolver_wrapper(
    df: DataFrame, resolver: Callable[[Iterable[T]], Iterator[tuple[T, V]]], api: str
) -> DataFrame:
    assert (
        df.columns == CACHE_COLUMNS[:1]
    ), f"The cache misses should consist of just one column named '{CACHE_COLUMNS[0]}'"
    ret = df.rdd.mapPartitions(resolver)
    # The order of columns must match the initial cache. One could enforce it, by having the initial cache (or its schema) as an input, but that would be inefficient.
    return ret.toDF(schema=df.schema.add(CACHE_COLUMNS[1], data_type=ArrayType(FloatType()))).withColumn(
        CACHE_COLUMNS[2], F.lit(api)
    )


@inject_object()
def lookup_from_cache(
    df: DataFrame,
    cache: DataFrame,
    api: str,
    primary_key: str,
    preprocessor: Callable[[DataFrame], DataFrame],
    new_col: str,
) -> DataFrame:
    cache = (
        limit_cache_to_results_from_api(cache, api=api)
        .transform(resolve_cache_duplicates, id_col=CACHE_COLUMNS[0])
        .withColumnsRenamed({CACHE_COLUMNS[0]: primary_key, CACHE_COLUMNS[1]: new_col})
    )
    return df.transform(preprocessor).join(cache, how="left", on=primary_key)


def limit_cache_to_results_from_api(df: DataFrame, api: str) -> DataFrame:
    """Filter the cache for results from a particular `api`.

    Note that the reason for filtering for results from the api is that you likely don't want to add results from a different API than the one you use to resolve cache misses."""
    return df.filter(df[CACHE_COLUMNS[2]] == api).drop(CACHE_COLUMNS[2])


def resolve_cache_duplicates(df: DataFrame, id_col: str) -> DataFrame:
    """Return a DataFrame where duplicates have been removed in a non-discriminatory fashion.

    Duplicates are detected by their `id_col`."""
    keys, distinct_keys = df.agg(F.count("*"), F.count_distinct(id_col)).first()
    if keys != distinct_keys:
        # Warnings can be converted to errors, making them more ideal here.
        warnings.warn(
            f"The cache contains duplicate keys. This is likely the result of a concurrent run and should be prevented. Continuing by non-discriminatory dropping duplicates…"
        )
        df = df.drop_duplicates([id_col])
    return df


def cached_api_enrichment_pipeline(
    input: str,
    cache: str,
    cache_out: str,
    cache_misses: str,
    primary_key: str,
    cache_miss_resolver: str,  # Callable[[str], Callable[[Iterable[T]], Iterator[V]]]
    api: str,
    preprocessor: str,  # Callable[[DataFrame], DataFrame],
    output: str,
    new_col: str,
) -> Pipeline:
    """Pipeline to enrich a dataframe using optionally cached API calls.

    The advantage to using this is that any identical API calls that have
    been made in other runs will have been cached, so that the enrichment
    process will complete faster.

    Note: ideally the preprocessor is a no-op, a simple pass-through (like
    `lambda x: x`). If you can shape your DataFrame in such a way that it
    doesn't need a custom preprocessor (e.g. by calling your own preprocessor
    in the previous node, at the end), it'll be computationally more efficient
    (meaning this subpipeline will complete faster, and thus saves the
    organization money).

    input: Kedro reference to a Spark DataFrame where you want to add a column
    `new_col` to.

    cache: Kedro reference to the Spark DataFrame that maps keys to values. The
    keys will be compared to the `primary_key` column of the DataFrame
    resulting from calling the `preprocessor` on the `input`. Aside from the
    key and value column, it also has a third column, named api, linking the
    keys to the API used at the time of the lookup.

    cache_out: same Kedro reference as cache, but allows circumventing circular
    references in Kedro. Its catalog entry should have the append-mode in its
    save_args.

    cache_misses: a Kedro reference to a SparkDataset that is used for temporary results.

    primary_key: name of the column that should be produced (or passed) by the
    preprocessor function. It is this column of values that will be used to
    check against the cache, or failing that, sent to the cache_miss_resolver.

    cache_miss_resolver: Kedro reference to a callable that will be used to
    look up any cache misses.

    api: Kedro parameter to restrict the cache to use results from this
    particular API

    preprocessor: Kedro reference to a callable that will preprocess the
    `input` such that it has a column `primary_key` which is used in the
    look-up process.

    output: Kedro reference to a dataset that is the input plus this new
    `new_col` containing the results from the enrichment with the cache/API.

    new_col: name of the column in which the values associated with the
    primary_key should appear.
    """

    common_inputs = {"df": input, "cache": cache, "api": api, "primary_key": primary_key, "preprocessor": preprocessor}
    nodes = [
        ArgoNode(
            name="derive_cache_misses",
            func=derive_cache_misses,
            inputs=common_inputs,
            outputs=cache_misses,
        ),
        ArgoNode(
            name="resolve_cache_misses",
            func=cache_miss_resolver_wrapper,
            inputs={"df": cache_misses, "resolver": cache_miss_resolver, "api": api},
            outputs=cache_out,
        ),
        ArgoNode(
            name="lookup_from_cache",
            func=lookup_from_cache,
            # By supplying the output of the previous node, which shares the same path as the starting cache, as an input, Kedro reloads the dataset, noticing now that it had more data.
            # Note that replacing the `cache_out` by `cache`, it is NOT reloaded, as `cache` is an input.
            inputs=common_inputs | {"cache": cache_out, "new_col": new_col},
            outputs=output,
        ),
    ]

    return pipeline(nodes)


def dummy_resolver(sequence: Collection[T], api) -> Iterator[tuple[T, V]]:
    seq = tuple(sequence)
    logging.debug(f"resolving, {seq}, using {api}")
    yield from zip((_[0] for _ in seq), [[5.0, 3.0]] * len(seq))


def pass_through(x):
    return x
