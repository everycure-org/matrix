"""Nodes for the data publication pipeline.

This pipeline publishes datasets to HuggingFace Hub for public access.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


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
