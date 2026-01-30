"""Polars-based validator check."""

import json

import polars as pl

from matrix_validator.checks import DELIMITED_BY_PIPES


def validate(df, column):
    """Validate Array to be delimited by pipes."""
    violations_df = df.select(
        [
            pl.when(~pl.col(column).str.contains(DELIMITED_BY_PIPES))
            .then(pl.col(column))
            .otherwise(pl.lit(None))
            .alias(f"invalid_delimited_by_pipes_{column}"),
        ]
    ).filter(pl.col(f"invalid_delimited_by_pipes_{column}").is_not_null())

    # Count total violations
    total_violations = len(violations_df)
    if total_violations == 0:
        return ""

    # Get unique violations and limit to 10 examples
    unique_violations = violations_df.unique().head(10)
    examples = unique_violations.get_column(f"invalid_delimited_by_pipes_{column}").to_list()

    # Create a summary report
    report = {"total_violations": total_violations, "unique_violations": len(unique_violations), "examples": examples}

    # Format output as a single JSON string
    result = {"error": {f"invalid_delimited_by_pipes_{column}_summary": report}}

    return json.dumps(result)
