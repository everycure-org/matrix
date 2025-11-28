from __future__ import annotations

from typing import Tuple

import pandas as pd  # type: ignore[import-not-found]
import polars as pl  # type: ignore[import-not-found]
from pyspark.sql import (
    DataFrame,  # type: ignore[import-not-found]
    SparkSession,  # type: ignore[import-not-found]
)
from pyspark.sql import functions as F  # type: ignore[import-not-found]


def generate_data_and_save_to_hf() -> Tuple[DataFrame, pd.DataFrame, pl.DataFrame]:
    """
    Generate knowledge graph edges data and return in three formats:
    Spark DataFrame, Pandas DataFrame, and Polars DataFrame.

    All three datasets will be saved to Hugging Face Hub by Kedro.

    Returns:
        Tuple of (spark_df, pandas_df, polars_df)
    """
    spark = SparkSession.builder.getOrCreate()

    base = spark.range(0, 100_000).withColumnRenamed("id", "idx")

    # A small set of example predicates (can be expanded later)
    predicates = [
        "biolink:related_to",
        "biolink:treats",
        "biolink:causes",
        "biolink:interacts_with",
        "biolink:associated_with",
        "biolink:contraindicated_for",
        "biolink:ameliorates",
        "biolink:exacerbates",
        "biolink:biomarker_for",
        "biolink:targets",
    ]

    pred_expr = F.expr(
        "case "
        + " ".join(f"when (idx % {len(predicates)}) = {i} then '{p}'" for i, p in enumerate(predicates))
        + f" else '{predicates[0]}' end"
    )

    spark_df = base.select(
        F.concat(F.lit("EC-DRUG-"), F.col("idx")).alias("subject"),
        pred_expr.alias("predicate"),
        F.concat(F.lit("EC-DISEASE-"), (F.col("idx") % F.lit(1000))).alias("object"),
        F.concat(F.lit("Edge "), F.col("idx")).alias("description"),
    )

    # Convert to pandas and polars
    pandas_df = spark_df.toPandas()
    polars_df = pl.from_pandas(pandas_df)

    return spark_df, pandas_df, polars_df


def read_hf_datasets_and_write_jsonl(
    spark_dataset: DataFrame, pandas_dataset: pd.DataFrame, polars_dataset: pl.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read three datasets from Hugging Face Hub and convert all to pandas DataFrames
    for JSONL writing.

    Args:
        spark_dataset: Spark dataset loaded from HF Hub
        pandas_dataset: Pandas dataset loaded from HF Hub
        polars_dataset: Polars dataset loaded from HF Hub

    Returns:
        Tuple of three pandas DataFrames for JSONL output
    """
    # Convert all to pandas for JSONL writing
    spark_pandas = spark_dataset.toPandas()
    pandas_df = pandas_dataset  # Already pandas
    polars_pandas = polars_dataset.to_pandas()

    return spark_pandas, pandas_df, polars_pandas
