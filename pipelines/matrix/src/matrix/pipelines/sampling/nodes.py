"""Nodes for extracting sample from the KG."""
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window


def sample_nodes(
    nodes=DataFrame,
    stratify_by: str = "category",
    sample_ratio: int = 0.2,
    seed: int = 42,
):
    """Doc string for extracting stratified samples from nodes.

    Args:
        nodes: DataFrame of nodes.
        stratify_by: column by which one should stratify.
        sample_ratio: size of subgraph.
        seed: for reproduciblity.
    """
    # Define the window partitioned by stratify_by and ordered by a random column
    window_spec = Window.partitionBy(stratify_by).orderBy(F.rand(seed))

    # Add the row_number column based on the window
    nodes_with_row_num = nodes.withColumn(
        "row_number", F.row_number().over(window_spec)
    )

    # Calculate the number of rows to sample per group
    total_nodes_per_category = (
        nodes.groupBy(stratify_by).count().withColumnRenamed("count", "total_count")
    )
    nodes_with_row_num = nodes_with_row_num.join(
        total_nodes_per_category, on=stratify_by, how="left"
    )

    # Filter to get the sample based on the sample_ratio (10% of each group)
    sampled_nodes = nodes_with_row_num.filter(
        F.col("row_number") <= F.col("total_count") * sample_ratio
    )

    # Drop the helper columns
    sampled_nodes = sampled_nodes.drop("row_number", "total_count")

    return sampled_nodes


def sample_edges(
    nodes, edges, stratify_by: str = "predicate", ratio: float = 0.1, seed: int = 42
):
    """Doc string for extracting stratified samples from nodes.

    Args:
        nodes: DataFrame of nodes post-subsample extraction.
        edges: DataFrame of edges.
        stratify_by: column by which one should stratify.
        ratio: size of subgraph.
        seed: for reproduciblity.
    """
    # Step 1: Alias the nodes DataFrame for source and target node joins
    nodes_alias_source = nodes.alias("source_nodes")
    nodes_alias_target = nodes.alias("target_nodes")

    # Step 2: Filter edges where both source and target nodes exist in the sampled node set
    sampled_edges = edges.join(
        nodes_alias_source, edges["subject"] == F.col("source_nodes.id"), how="inner"
    ).join(nodes_alias_target, edges["object"] == F.col("target_nodes.id"), how="inner")

    # Select only necessary columns and rename them if needed to avoid conflicts
    sampled_edges = sampled_edges.select(
        F.col("subject"),
        F.col("predicate"),
        F.col("object"),
        F.col("knowledge_source"),
        F.col("kg_sources"),
    )

    # Step 3: (Optional) Stratify the edges by edge type if needed
    if stratify_by:
        window_spec = Window.partitionBy(stratify_by).orderBy(F.rand(seed))
        sampled_edges_with_row_num = sampled_edges.withColumn(
            "row_number", F.row_number().over(window_spec)
        )

        # Calculate the number of rows to sample per group
        total_edges_per_category = (
            sampled_edges.groupBy(stratify_by)
            .count()
            .withColumnRenamed("count", "total_count")
        )
        sampled_edges_with_row_num = sampled_edges_with_row_num.join(
            total_edges_per_category, on=stratify_by, how="left"
        )

        # Filter edges to get a sample based on the ratio
        sampled_edges = sampled_edges_with_row_num.filter(
            F.col("row_number") <= F.col("total_count") * ratio
        )

        # Drop helper columns
        sampled_edges = sampled_edges.drop("row_number", "total_count")

    # Step 4: Return the final sampled edges
    return sampled_edges
