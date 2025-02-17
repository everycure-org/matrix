from typing import Any, Dict, List, Optional

import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline
from matrix.inject import inject_object
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


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
