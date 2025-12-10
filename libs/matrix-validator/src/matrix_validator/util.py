"""Utilities for the matrix validator."""

from importlib import resources as il_resources

import polars as pl
import yaml
from biolink_model import schema
from matrix_schema import schema as matrix_schema
from yaml import SafeLoader


def read_tsv_as_strings(file_path):
    """Read a TSV file with all columns interpreted as strings."""
    return pl.scan_csv(
        file_path,
        separator="\t",
        infer_schema_length=0,  # Avoid inferring any schema
    )


def get_biolink_model_knowledge_level_keys():
    """Get biolink model knowledge_level keys."""
    # Use files() instead of read_text as per deprecation warning
    with il_resources.files(schema).joinpath("biolink_model.yaml").open("r") as f:
        bl_model_data = list(yaml.load_all(f.read(), Loader=SafeLoader))
    return list(bl_model_data[0]["enums"]["KnowledgeLevelEnum"]["permissible_values"].keys())


def get_biolink_model_agent_type_keys():
    """Get biolink model agent_type keys."""
    # Use files() instead of read_text as per deprecation warning
    with il_resources.files(schema).joinpath("biolink_model.yaml").open("r") as f:
        bl_model_data = list(yaml.load_all(f.read(), Loader=SafeLoader))
    return list(bl_model_data[0]["enums"]["AgentTypeEnum"]["permissible_values"].keys())


def get_valid_edge_types() -> list[dict[str, str]]:
    """Get valid edge types from the matrix schema."""
    # Use files() instead of read_text as per deprecation warning
    with il_resources.files(matrix_schema).joinpath("valid_biolink_edge_types.tsv").open("r") as f:
        valid_edge_string = f.read()

    # get the keys from the first line
    keys = valid_edge_string.splitlines()[0].split("\t")
    # get the values from the rest of the lines
    values = [line.split("\t") for line in valid_edge_string.splitlines()[1:]]
    # create a list of dictionaries
    return [dict(zip(keys, value, strict=False)) for value in values if len(value) == len(keys)]


def analyze_edge_types(nodes_df: pl.DataFrame, edges_df: pl.DataFrame) -> pl.DataFrame:
    """
    Analyze edge types from the data being validated.

    This function joins the edge table with the node table twice (once for subject, once for object)
    to get the subject_category and object_category for each edge. It then groups by
    subject_category, predicate, and object_category to count occurrences, and joins with
    the valid edge types to determine which edge types are valid according to the biolink model.

    Args:
        nodes_df: Polars DataFrame containing node data with at least 'id' and 'category' columns
        edges_df: Polars DataFrame containing edge data with at least 'subject', 'predicate', and 'object' columns

    Returns:
        Polars DataFrame with columns: subject_category, predicate, object_category, valid, count

    """
    # Make sure required columns are present
    required_node_cols = ["id", "category"]
    required_edge_cols = ["subject", "predicate", "object"]

    for col in required_node_cols:
        if col not in nodes_df.columns:
            raise ValueError(f"Required column '{col}' not found in nodes_df")

    for col in required_edge_cols:
        if col not in edges_df.columns:
            raise ValueError(f"Required column '{col}' not found in edges_df")

    # Convert category column to string if needed
    if nodes_df["category"].dtype != pl.Utf8:
        nodes_df = nodes_df.with_columns(pl.col("category").cast(pl.Utf8))

    # Join edges with nodes to get subject_category
    edges_with_subject = edges_df.join(nodes_df.select(["id", "category"]), left_on="subject", right_on="id", how="left").rename(
        {"category": "subject_category"}
    )

    # Join with nodes again to get object_category
    edges_with_categories = edges_with_subject.join(
        nodes_df.select(["id", "category"]), left_on="object", right_on="id", how="left"
    ).rename({"category": "object_category"})

    # Group by subject_category, predicate, object_category and count
    edge_type_counts = edges_with_categories.group_by(["subject_category", "predicate", "object_category"]).agg(count=pl.len())

    # Get valid edge types from biolink model
    valid_edge_types = get_valid_edge_types()

    # Check if each edge type in our data is valid
    edge_type_analysis = edge_type_counts.with_columns(
        valid=pl.struct(["subject_category", "predicate", "object_category"]).is_in(
            pl.Series(
                [
                    {"subject_category": et["subject_category"], "predicate": et["predicate"], "object_category": et["object_category"]}
                    for et in valid_edge_types
                ]
            )
        )
    )

    # Handle potential nulls in categories (when nodes referenced in edges don't exist)
    edge_type_analysis = edge_type_analysis.with_columns(
        [
            pl.when(pl.col("subject_category").is_null())
            .then(pl.lit("unknown"))
            .otherwise(pl.col("subject_category"))
            .alias("subject_category"),
            pl.when(pl.col("object_category").is_null())
            .then(pl.lit("unknown"))
            .otherwise(pl.col("object_category"))
            .alias("object_category"),
        ]
    )

    # Sort by count (descending)
    return edge_type_analysis.sort("count", descending=True)
