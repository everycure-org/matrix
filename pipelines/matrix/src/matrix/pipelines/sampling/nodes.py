"""Nodes to sample datasets."""

from typing import Tuple, Dict
from pyspark.sql import DataFrame


def sample_datasets(**kwargs) -> Tuple[DataFrame]:
    """Samples datasets to a fixed row count and returns them."""
    # sample each dataframe to the row count
    sampled_dfs = []
    ROW_COUNT = 10  # FUTURE make this a config value
    for df in kwargs.values():
        sampled_dfs.append(df.limit(ROW_COUNT))
    dfs = tuple(sampled_dfs)
    print(_generate_markdown_report(kwargs))
    return dfs


def _generate_markdown_report(kwargs: Dict[str, DataFrame]) -> str:
    """Generates a markdown report of the sampled datasets."""
    buffer = []
    buffer.append("# Sampling Report")

    for key, df in kwargs.items():
        buffer.append(f"## {key}")
        # first log all datatypes
        buffer.append("Data types:\n")
        buffer.append("| Column | Data Type |")
        buffer.append("|--------|-----------|")
        for col, dtype in df.dtypes:
            buffer.append(f"| {col} | {dtype} |")
        buffer.append("\n")

        # then log the first 10 rows
        buffer.append("First 10 rows:\n")
        buffer.append(df.limit(10).toPandas().to_markdown(index=False))
        buffer.append("\n")

    return "\n".join(buffer)
