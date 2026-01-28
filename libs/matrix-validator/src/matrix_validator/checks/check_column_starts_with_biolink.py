"""Polars-based validator check."""

import json

import polars as pl

from matrix_validator.checks import STARTS_WITH_BIOLINK_REGEX


def validate(df, column):
    """Validate column to start with biolink:."""
    violations_df = df.select(
        [
            pl.when(~pl.col(column).str.contains(STARTS_WITH_BIOLINK_REGEX))
            .then(pl.col(column))
            .otherwise(pl.lit(None))
            .alias(f"invalid_starts_with_biolink_{column}"),
        ]
    ).filter(pl.col(f"invalid_starts_with_biolink_{column}").is_not_null())

    # Count total violations
    total_violations = len(violations_df)
    if total_violations == 0:
        return ""

    # Get unique violations and limit to 10 examples
    unique_violations = violations_df.unique().head(10)
    examples = unique_violations.get_column(f"invalid_starts_with_biolink_{column}").to_list()

    # Create a summary report
    report = {"total_violations": total_violations, "unique_violations": len(unique_violations), "examples": examples}

    # Format output as a single JSON string
    result = {"error": {f"invalid_starts_with_biolink_{column}_summary": report}}

    return json.dumps(result)
