"""Nodes for the data publication pipeline.

This pipeline publishes datasets to HuggingFace Hub for public access.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import DataFrame


def publish_dataset_to_hf(data: DataFrame) -> DataFrame:
    """
    Publish a dataset to HuggingFace Hub.

    This node simply passes through the data; the actual HuggingFace upload
    is handled by Kedro's HFIterableDataset in the catalog configuration.

    Args:
        data: Spark DataFrame to publish

    Returns:
        The same DataFrame (passthrough for pipeline continuity)
    """
    # The dataset publishing happens via the catalog configuration
    # This node just passes the data through
    return data


def verify_published_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """
    Verify that a dataset was successfully published by reading it back.

    Args:
        data: pandas DataFrame loaded from HuggingFace Hub

    Returns:
        Dictionary with verification statistics
    """
    count = len(data)
    columns = list(data.columns)

    stats = {
        "row_count": int(count),
        "column_count": len(columns),
        "columns": columns,
    }

    return stats
