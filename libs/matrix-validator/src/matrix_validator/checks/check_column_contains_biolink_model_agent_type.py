"""Polars-based validator check."""

import json

import polars as pl
from polars import DataFrame


def validate(df: DataFrame, column, bm_agent_types: list):
    """Validate contains Biolink Model Agent Type."""
    violations_df = df.select(
        [
            pl.when(~pl.col(column).str.contains_any(bm_agent_types))
            .then(pl.col(column))
            .otherwise(pl.lit(None))
            .alias(f"invalid_contains_biolink_model_agent_type_{column}"),
        ]
    ).filter(pl.col(f"invalid_contains_biolink_model_agent_type_{column}").is_not_null())

    # Count total violations
    total_violations = len(violations_df)
    if total_violations == 0:
        return ""

    # Get unique violations and limit to 10 examples
    unique_violations = violations_df.unique()
    examples = unique_violations.get_column(f"invalid_contains_biolink_model_agent_type_{column}").head(10).to_list()

    # Create a summary report
    report = {
        "total_violations": total_violations,
        "unique_violations": len(unique_violations),
        "examples": examples,
        "valid_values": bm_agent_types,
    }

    # Format output as a single JSON string
    result = {"error": {f"invalid_contains_biolink_model_agent_type_{column}_summary": report}}

    return json.dumps(result)
