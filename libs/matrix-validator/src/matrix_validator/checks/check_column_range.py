"""Polars-based validator check."""

import json
import logging

import polars as pl
from polars import DataFrame

logger = logging.getLogger(__name__)


def validate(df: DataFrame, column, min, max):
    """Validate range."""
    violations_df = df.select(
        [
            pl.when(~pl.col(column).is_between(min, max))
            .then(pl.col(column))
            .otherwise(pl.lit(None))
            .alias(f"invalid_column_range_{column}"),
        ]
    ).filter(pl.col(f"invalid_column_range_{column}").is_not_null())

    # Count total violations
    total_violations = len(violations_df)
    if total_violations == 0:
        return ""

    # Get unique violations and limit to 10 examples
    unique_violations = violations_df.unique()
    examples = unique_violations.get_column(f"invalid_column_range_{column}").head(10).to_list()

    # Create a summary report
    report = {
        "error": {
            "check": f"invalid_column_range_{column}_summary",
            "total_violations": total_violations,
            "unique_violations": len(unique_violations),
            "range": f"{min}..{max}",
            "examples": examples,
        }
    }

    return json.dumps(report)
