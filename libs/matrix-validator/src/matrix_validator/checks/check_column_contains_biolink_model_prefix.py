"""Polars-based validator check."""

import json

import polars as pl


def validate(df, column, bm_prefixes: list):
    """Validate contains Biolink Model prefix."""
    violations_df = df.select(
        [
            pl.when(~pl.col(column).str.contains_any(bm_prefixes, ascii_case_insensitive=True))
            .then(pl.col(column))
            .otherwise(pl.lit(None))
            .alias(f"invalid_contains_biolink_model_prefix_{column}"),
        ]
    ).filter(pl.col(f"invalid_contains_biolink_model_prefix_{column}").is_not_null())

    # Count total violations and get unique examples
    total_violations = len(violations_df)
    if total_violations == 0:
        return ""

    # Group violations by prefix (extract prefix from before the colon)
    with_prefix = violations_df.with_columns(
        pl.col(f"invalid_contains_biolink_model_prefix_{column}").str.split(":").list.first().alias("prefix")
    )

    # Group by prefix and count
    prefix_counts = (
        with_prefix.group_by("prefix")
        .agg(pl.count().alias("count"), pl.col(f"invalid_contains_biolink_model_prefix_{column}").head(3).alias("examples"))
        .sort("count", descending=True)
    )

    # Create a summary report
    report = {"total_violations": total_violations, "prefix_violations": prefix_counts.to_dicts()}

    # Format output as a single JSON string
    result = {"error": {f"invalid_contains_biolink_model_prefix_{column}_summary": report}}

    return json.dumps(result)
