"""Perturbation nodes for edge rewiring experiments."""

import logging
from typing import Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def perturb_edges_rewire(
    unified_edges: DataFrame,
    unified_nodes: DataFrame,
    perturbation_rate: float,
    random_seed: int = 42,
    strategy: str = "category_stratified",
) -> DataFrame:
    """
    Rewire a percentage of edges to different nodes of same category.

    Args:
        unified_edges: Complete unified edge list (Spark DataFrame)
        unified_nodes: Complete unified node list with categories
        perturbation_rate: Fraction of edges to rewire (0.01, 0.05, 0.20, 0.50)
        random_seed: For reproducibility
        strategy: "category_stratified" - maintain node category distribution

    Returns:
        Rewired edge DataFrame with same schema as input

    Algorithm:
        1. Join edges with nodes to get object categories
        2. Sample perturbation_rate of edges stratified by object category
        3. For each category, create a pool of alternative targets
        4. Randomly reassign object_ids within each category
        5. Union rewired edges with non-perturbed edges
    """

    logger.info(f"Starting edge rewiring with {perturbation_rate:.1%} perturbation rate")
    logger.info(f"Random seed: {random_seed}, Strategy: {strategy}")

    # Get initial counts
    total_edges = unified_edges.count()
    logger.info(f"Total edges before perturbation: {total_edges:,}")

    # Step 1: Enrich edges with object node categories
    logger.info("Step 1: Enriching edges with object node categories")
    edges_with_categories = (
        unified_edges.join(
            unified_nodes.select("id", "category").alias("obj_nodes"), F.col("object") == F.col("obj_nodes.id"), "inner"
        )
        .select(
            "subject",
            "predicate",
            "object",
            F.col("obj_nodes.category").alias("object_category"),
            # Include all other edge columns dynamically
            *[F.col(c) for c in unified_edges.columns if c not in ["subject", "predicate", "object"]],
        )
        .cache()
    )

    edges_with_cats_count = edges_with_categories.count()
    logger.info(f"Edges with categories: {edges_with_cats_count:,}")

    # Step 2: Sample edges for perturbation, stratified by object category
    logger.info("Step 2: Sampling edges for perturbation")
    edges_for_perturbation = (
        edges_with_categories.withColumn("random_val", F.rand(seed=random_seed))
        .filter(F.col("random_val") < perturbation_rate)
        .withColumn("edge_id", F.monotonically_increasing_id())
        .cache()
    )

    selected_for_perturbation = edges_for_perturbation.count()
    logger.info(f"Edges selected for perturbation: {selected_for_perturbation:,}")

    if selected_for_perturbation == 0:
        logger.warning("No edges selected for perturbation. Returning original edges.")
        edges_with_categories.unpersist()
        edges_for_perturbation.unpersist()
        return unified_edges

    # Step 3: Create pools of alternative targets by category
    logger.info("Step 3: Creating target pools by category")
    target_pools_by_category = (
        unified_nodes.select("id", "category")
        .groupBy("category")
        .agg(F.collect_list("id").alias("candidate_targets"))
        .filter(
            F.size(F.col("candidate_targets")) > 1  # Need at least 2 nodes to rewire
        )
        .cache()
    )

    category_count = target_pools_by_category.count()
    logger.info(f"Categories available for rewiring: {category_count}")

    # Step 4: Rewire edges within each category
    logger.info("Step 4: Rewiring edges within categories")
    edges_to_rewire = edges_for_perturbation.join(
        F.broadcast(target_pools_by_category), F.col("object_category") == F.col("category"), "inner"
    )

    # Create random reassignments within each category
    rewired_edges = (
        edges_to_rewire.withColumn(
            "random_index", (F.rand(seed=random_seed + 1) * F.size(F.col("candidate_targets"))).cast("int")
        )
        .withColumn("new_object", F.expr("candidate_targets[random_index]"))
        .filter(
            F.col("new_object") != F.col("object")  # Ensure we actually change the target
        )
        .select(
            F.col("subject"),
            F.col("predicate"),
            F.col("new_object").alias("object"),
            # Include all other original edge columns
            *[F.col(c) for c in unified_edges.columns if c not in ["subject", "predicate", "object"]],
        )
    )

    rewired_count = rewired_edges.count()
    logger.info(f"Successfully rewired edges: {rewired_count:,}")

    # Step 5: Get edges that were NOT selected for perturbation
    logger.info("Step 5: Collecting unperturbed edges")
    unperturbed_edges = edges_with_categories.join(
        edges_for_perturbation.select("subject", "predicate", "object", "edge_id"),
        ["subject", "predicate", "object"],
        "left_anti",
    ).select(
        *[F.col(c) for c in unified_edges.columns]  # Original schema
    )

    unperturbed_count = unperturbed_edges.count()
    logger.info(f"Unperturbed edges: {unperturbed_count:,}")

    # Step 6: Union rewired and unperturbed edges
    logger.info("Step 6: Combining rewired and unperturbed edges")
    final_edges = unperturbed_edges.unionByName(rewired_edges, allowMissingColumns=True)

    # Final count check
    final_count = final_edges.count()
    logger.info(f"Final edge count: {final_count:,}")

    # Log statistics
    actual_perturbation_rate = rewired_count / total_edges if total_edges > 0 else 0
    logger.info(f"Actual perturbation rate: {actual_perturbation_rate:.1%}")

    # Cleanup cached DataFrames
    edges_with_categories.unpersist()
    edges_for_perturbation.unpersist()
    target_pools_by_category.unpersist()

    return final_edges.cache()


def passthrough_edges(edges: DataFrame) -> DataFrame:
    """
    Passthrough function for when perturbation is disabled.

    Args:
        edges: Input edges DataFrame

    Returns:
        Same edges DataFrame unchanged
    """
    logger.info("Perturbation disabled - passing through edges unchanged")
    return edges


def log_rewiring_stats(
    edges_before: DataFrame, edges_after: DataFrame, perturbation_rate: float, unified_nodes: Optional[DataFrame] = None
) -> None:
    """
    Log statistics about the rewiring process.

    Args:
        edges_before: Edges before perturbation
        edges_after: Edges after perturbation
        perturbation_rate: Target perturbation rate
        unified_nodes: Optional nodes for category analysis
    """

    total_before = edges_before.count()
    total_after = edges_after.count()
    expected_rewired = int(total_before * perturbation_rate)

    logger.info("=" * 50)
    logger.info("REWIRING STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Total edges before: {total_before:,}")
    logger.info(f"Total edges after: {total_after:,}")
    logger.info(f"Expected rewired edges: {expected_rewired:,}")
    logger.info(f"Target perturbation rate: {perturbation_rate:.1%}")

    if abs(total_before - total_after) > 1000:  # Allow small differences
        logger.warning(f"Edge count changed significantly: {total_after - total_before:+,}")

    # Basic category distribution check if nodes provided
    if unified_nodes is not None and total_before < 1_000_000:
        try:
            category_stats_before = (
                edges_before.join(unified_nodes.select("id", "category"), F.col("object") == F.col("id"))
                .groupBy("category")
                .count()
                .orderBy(F.desc("count"))
                .limit(10)
                .collect()
            )

            logger.info("Top 10 object categories (before):")
            for row in category_stats_before:
                logger.info(f"  {row['category']}: {row['count']:,}")

        except Exception as e:
            logger.warning(f"Could not compute category statistics: {e}")

    logger.info("=" * 50)
