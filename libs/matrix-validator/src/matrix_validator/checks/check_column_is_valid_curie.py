"""Polars-based validator check."""

import json

import polars as pl

from matrix_validator.checks import CURIE_REGEX


def validate(df, column):
    """Validate column to be a valid CURIE."""
    violations_df = df.select(
        [
            pl.when(~pl.col(column).str.contains(CURIE_REGEX))
            .then(pl.col(column))
            .otherwise(pl.lit(None))
            .alias(f"invalid_curie_{column}"),
        ]
    ).filter(pl.col(f"invalid_curie_{column}").is_not_null())

    # Count total violations
    total_violations = len(violations_df)
    if total_violations == 0:
        return ""

    # Get unique violations and limit to 10 examples
    unique_violations = violations_df.unique().head(10)
    examples = unique_violations.get_column(f"invalid_curie_{column}").to_list()

    # Create a summary report
    report = {"total_violations": total_violations, "unique_violations": len(unique_violations), "examples": examples}

    # Format output as a single JSON string
    result = {"error": {f"invalid_curie_{column}_summary": report}}

    return json.dumps(result)
