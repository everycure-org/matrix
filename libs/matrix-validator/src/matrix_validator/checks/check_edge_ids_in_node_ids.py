"""Polars-based validator check."""

import json

import polars as pl


def validate(df, edge_ids: list, column: str):
    """Validate that all edge subject/object IDs exist in Nodes."""
    # Create a dataframe from edge IDs with a column name matching the nodes dataframe
    edge_df = pl.DataFrame({column: edge_ids})

    # Use an anti-join to find missing edge IDs (edge IDs not in node IDs)
    # This is highly efficient as it uses hashing internally
    missing_df = edge_df.join(df.select(pl.col(column)), on=column, how="anti")

    # If no violations, return early
    if missing_df.height == 0:
        return ""

    # Rename the column for consistency with the rest of the function
    violations_df = missing_df.rename({column: "invalid_edge_ids_in_node_ids"})

    # For very large results, sample for reporting
    if violations_df.height > 1000:
        violations_df = violations_df.sample(1000, seed=42)

    # Count total violations (from the original anti-join)
    total_violations = missing_df.height

    # Group violations by prefix (extract prefix from before the colon)
    with_prefix = violations_df.with_columns(pl.col("invalid_edge_ids_in_node_ids").str.split(":").list.first().alias("prefix"))

    # Group by prefix and count
    prefix_counts = (
        with_prefix.group_by("prefix")
        .agg(pl.len().alias("count"), pl.col("invalid_edge_ids_in_node_ids").head(3).alias("examples"))
        .sort("count", descending=True)
    )

    # Create a summary report

    # Format output as a single JSON string
    result = {
        "error": {
            "invalid_edge_ids_in_node_ids_summary": {"total_violations": total_violations, "prefix_violations": prefix_counts.to_dicts()}
        }
    }

    return json.dumps(result)
