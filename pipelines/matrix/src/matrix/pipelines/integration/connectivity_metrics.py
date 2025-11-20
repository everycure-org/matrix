"""
Connectivity metrics for knowledge graph analysis.

This module contains functions for computing connected components and analyzing
graph connectivity using various implementations (GraphFrames, grape, rustworkx).
"""

import gc
import logging
import time
from collections import Counter
from typing import Dict

import pyspark.sql as ps
import pyspark.sql.functions as F

logger = logging.getLogger(__name__)


def _simplify_graph_for_connectivity(edges: ps.DataFrame) -> ps.DataFrame:
    """
    Helper function to simplify a graph by removing duplicate edges between nodes.

    For connectivity analysis, we only care whether two nodes are connected,
    not the edge metadata (predicate, knowledge source, etc.). This function
    creates a simplified edge list with unique (subject, object) pairs,
    making connectivity calculations more efficient and accurate.

    Args:
        edges: DataFrame with at minimum 'subject' and 'object' columns

    Returns:
        DataFrame with unique (subject, object) pairs

    Example:
        Input edges:
            subject='A', object='B', predicate='treats'
            subject='A', object='B', predicate='affects'
            subject='A', object='C', predicate='treats'

        Output edges:
            subject='A', object='B'
            subject='A', object='C'
    """
    original_count = edges.count()
    logger.info(f"Simplifying graph: {original_count:,} edges before deduplication")

    # Filter self-loops and keep only unique (subject, object) pairs
    simplified = (
        edges.select("subject", "object")
        .filter(F.col("subject") != F.col("object"))  # Remove self-loops
        .distinct()
    )

    simplified_count = simplified.count()
    reduction_pct = 100 * (1 - simplified_count / original_count) if original_count > 0 else 0

    logger.info(
        f"Graph simplified: {simplified_count:,} unique edges ({reduction_pct:.1f}% reduction from {original_count:,})"
    )

    return simplified


def compute_connected_components_graphframes(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    """
    Compute connected components using GraphFrames (Spark-native).

    GraphFrames is a graph processing library built on Apache Spark DataFrames.
    It leverages Spark's distributed computing for scalability on large graphs.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        DataFrame with component statistics:
            - component_id: Unique identifier for each component
            - component_size: Number of nodes in the component
            - num_components: Total number of connected components
    """
    from graphframes import GraphFrame

    logger.info("Computing connected components using GraphFrames...")

    # Prepare nodes for GraphFrame (needs 'id' column)
    # Note: nodes are already deduplicated from union_and_deduplicate_nodes
    gf_nodes = nodes.select(F.col("id"))

    # Prepare edges for GraphFrame (needs 'src' and 'dst' columns)
    gf_edges = edges.select(F.col("subject").alias("src"), F.col("object").alias("dst"))

    # Create GraphFrame
    graph = GraphFrame(gf_nodes, gf_edges)

    # Set checkpoint directory (required for connected components)
    spark = nodes.sparkSession
    checkpoint_dir = "/data/checkpoints/graphframes"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    # Compute connected components
    components = graph.connectedComponents()

    # Compute component sizes
    component_stats = (
        components.groupBy("component")
        .agg(F.count("*").alias("component_size"))
        .withColumnRenamed("component", "component_id")
    )

    # Add total component count
    num_components = component_stats.count()
    component_stats = component_stats.withColumn("num_components", F.lit(num_components))

    # Sort by component size descending
    component_stats = component_stats.orderBy(F.desc("component_size"))

    logger.info(
        f"Found {num_components:,} connected components using GraphFrames. "
        f"Largest component has {component_stats.first()['component_size']:,} nodes."
    )

    return component_stats


def compute_connected_components_grape(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    """
    Compute connected components using grape (Rust-based, ultra-fast).

    grape/ensmallen is a Rust-based graph processing library designed for
    billion-scale graphs with ultra-fast performance.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        DataFrame with component statistics:
            - component_id: Unique identifier for each component
            - component_size: Number of nodes in the component
            - num_components: Total number of connected components
    """
    from ensmallen import GraphBuilder

    logger.info("Computing connected components using grape...")

    # Build graph using GraphBuilder
    builder = GraphBuilder()
    builder.set_directed(False)
    builder.set_name("Connectivity_Graph")

    # Add all nodes
    # Note: nodes are already deduplicated from union_and_deduplicate_nodes
    logger.info("Collecting nodes for grape...")
    nodes_list = [row.id for row in nodes.select("id").collect()]
    logger.info(f"Collected {len(nodes_list):,} nodes")

    for node_id in nodes_list:
        builder.add_node(name=node_id)

    # Collect edges
    logger.info("Collecting edges for grape GraphBuilder...")
    edges_list = edges.select("subject", "object").collect()
    logger.info(f"Collected {len(edges_list):,} edges. Adding to graph...")

    edge_count = 0
    for row in edges_list:
        builder.add_edge(src=row.subject, dst=row.object)
        edge_count += 1
        if edge_count % 10_000_000 == 0:
            logger.info(f"Added {edge_count:,} edges...")

    logger.info(f"Building grape graph with {len(nodes_list):,} nodes and {edge_count:,} edges...")
    graph = builder.build()

    # Compute connected components using grape's API
    logger.info("Computing connected components with grape...")
    # connected_components() returns: (component_per_node, num_components, min_size, max_size)
    component_per_node, num_components, min_size, max_size = graph.get_connected_components()

    # Count component sizes
    component_sizes = Counter(component_per_node)

    # Create Spark DataFrame with component statistics
    # Convert numpy types to Python int for Spark compatibility
    spark = nodes.sparkSession
    component_data = [
        {"component_id": int(comp_id), "component_size": int(size), "num_components": int(num_components)}
        for comp_id, size in component_sizes.items()
    ]

    component_stats = spark.createDataFrame(component_data).orderBy(F.desc("component_size"))

    logger.info(
        f"Found {num_components:,} connected components using grape. "
        f"Largest component has {max(component_sizes.values()):,} nodes."
    )

    return component_stats


def compute_connected_components_rustworkx(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    """
    Compute connected components using rustworkx (Rust-based, high-performance).

    rustworkx is a high-performance graph library written in Rust with Python bindings,
    designed as a faster alternative to NetworkX.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        DataFrame with component statistics:
            - component_id: Unique identifier for each component
            - component_size: Number of nodes in the component
            - num_components: Total number of connected components
    """
    import rustworkx as rx

    logger.info("Computing connected components using rustworkx...")

    # --- Collect nodes ---
    logger.info("Collecting nodes for rustworkx...")
    nodes_list = [row.id for row in nodes.select("id").collect()]
    logger.info(f"Collected {len(nodes_list):,} nodes")

    # Create graph & add all nodes at once
    graph = rx.PyGraph()
    node_indices = graph.add_nodes_from(nodes_list)

    # Build mapping id -> index
    id_to_index = {node_id: idx for node_id, idx in zip(nodes_list, node_indices)}

    # --- Collect edges ---
    logger.info("Collecting edges for rustworkx...")
    edges_list = edges.select("subject", "object").collect()
    num_edges = len(edges_list)
    logger.info(f"Collected {num_edges:,} edges")

    # Build edge list for add_edges_from
    logger.info("Preparing edge list for bulk insertion...")
    edge_tuples = [
        (id_to_index[row.subject], id_to_index[row.object], None)
        for row in edges_list
        if row.subject in id_to_index and row.object in id_to_index
    ]

    logger.info(f"Bulk inserting {len(edge_tuples):,} edges...")
    graph.add_edges_from(edge_tuples)

    logger.info(f"Built rustworkx graph with {graph.num_nodes():,} nodes and {graph.num_edges():,} edges")

    # --- Connected components ---
    logger.info("Computing connected components with rustworkx...")
    components = rx.connected_components(graph)
    num_components = len(components)

    # --- Output statistics ---
    spark = nodes.sparkSession
    component_stats = spark.createDataFrame(
        [
            {"component_id": i, "component_size": len(c), "num_components": num_components}
            for i, c in enumerate(components)
        ]
    ).orderBy(F.desc("component_size"))

    largest_component_size = max((len(c) for c in components), default=0)

    logger.info(
        f"Found {num_components:,} connected components using rustworkx. "
        f"Largest component has {largest_component_size:,} nodes."
    )

    return component_stats


def compute_connected_components(nodes: ps.DataFrame, edges: ps.DataFrame, algorithm: str) -> ps.DataFrame:
    """
    Compute connected components using specified algorithm or benchmark all.

    This function first simplifies the graph by removing duplicate edges and
    self-loops, then dispatches to the appropriate connected components algorithm
    based on the algorithm parameter. In benchmark mode, it runs all three
    implementations, compares their performance (logged only, not saved), and
    returns results from the first algorithm.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (unified edges)
        algorithm: One of "graphframes", "grape", "rustworkx", or "benchmark"

    Returns:
        DataFrame with component statistics (from selected algorithm or first in benchmark)

    Raises:
        ValueError: If algorithm is not one of the supported options
    """
    valid_algorithms = ["graphframes", "grape", "rustworkx", "benchmark"]
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of: {', '.join(valid_algorithms)}")

    logger.info(f"Computing connected components with algorithm: {algorithm}")

    # Simplify graph for connectivity analysis
    simplified_edges = _simplify_graph_for_connectivity(edges)

    if algorithm == "benchmark":
        return _run_benchmark(nodes, simplified_edges)
    else:
        # Run single algorithm
        if algorithm == "graphframes":
            return compute_connected_components_graphframes(nodes, simplified_edges)
        elif algorithm == "grape":
            return compute_connected_components_grape(nodes, simplified_edges)
        elif algorithm == "rustworkx":
            return compute_connected_components_rustworkx(nodes, simplified_edges)


def _run_benchmark(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    """
    Run all three connected components algorithms and compare performance.

    Benchmark results are logged but not saved. This is intended for testing
    and algorithm selection purposes only.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        DataFrame with component statistics from the first algorithm (graphframes)
    """
    logger.info("=" * 80)
    logger.info("BENCHMARK MODE: Running all three connected components algorithms")
    logger.info("=" * 80)

    benchmark_results = []
    algorithm_outputs = {}

    # Run each algorithm and track performance
    algorithms = [
        ("graphframes", compute_connected_components_graphframes),
        ("grape", compute_connected_components_grape),
        ("rustworkx", compute_connected_components_rustworkx),
    ]

    for algo_name, algo_func in algorithms:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running {algo_name.upper()}...")
        logger.info(f"{'=' * 80}")

        start_time = time.time()
        results = algo_func(nodes, edges)
        end_time = time.time()

        execution_time = end_time - start_time

        # Get metrics from results
        first_row = results.first()
        num_components = first_row["num_components"]
        largest_component = first_row["component_size"]

        benchmark_results.append(
            {
                "algorithm": algo_name,
                "execution_time_seconds": round(execution_time, 2),
                "num_components": num_components,
                "largest_component_size": largest_component,
            }
        )

        algorithm_outputs[algo_name] = results

        logger.info(f"{algo_name.upper()} completed in {execution_time:.2f} seconds")

        # Force garbage collection between algorithms to free memory
        logger.info("Running garbage collection to free memory...")
        gc.collect()
        logger.info("Garbage collection complete")

    # Verify all algorithms produced same results
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFICATION: Checking consistency across algorithms")
    logger.info(f"{'=' * 80}")

    ref_components = benchmark_results[0]["num_components"]
    ref_largest = benchmark_results[0]["largest_component_size"]

    all_match = True
    for result in benchmark_results[1:]:
        if result["num_components"] != ref_components or result["largest_component_size"] != ref_largest:
            logger.warning(
                f"MISMATCH: {result['algorithm']} produced different results! "
                f"Components: {result['num_components']} vs {ref_components}, "
                f"Largest: {result['largest_component_size']} vs {ref_largest}"
            )
            all_match = False

    if all_match:
        logger.info("✓ All algorithms produced identical results")
        logger.info(f"  - Number of components: {ref_components:,}")
        logger.info(f"  - Largest component size: {ref_largest:,}")
    else:
        logger.error("✗ Algorithms produced DIFFERENT results - investigation needed!")

    # Create benchmark stats DataFrame
    logger.info(f"\n{'=' * 80}")
    logger.info("PERFORMANCE COMPARISON")
    logger.info(f"{'=' * 80}")

    spark = nodes.sparkSession
    benchmark_df = spark.createDataFrame(benchmark_results).orderBy("execution_time_seconds")

    # Log performance comparison
    for row in benchmark_df.collect():
        logger.info(
            f"{row['algorithm']:12s}: {row['execution_time_seconds']:8.2f}s "
            f"| Components: {row['num_components']:,} "
            f"| Largest: {row['largest_component_size']:,}"
        )

    # Determine fastest algorithm
    fastest = benchmark_df.first()
    logger.info(f"\n Fastest algorithm: {fastest['algorithm']} ({fastest['execution_time_seconds']:.2f}s)")

    logger.info(f"{'=' * 80}\n")

    # Return results from first algorithm (all should produce identical results)
    # Benchmark stats are logged above but not saved
    return algorithm_outputs["graphframes"]


def compute_core_connectivity_metrics(
    nodes: ps.DataFrame,
    core_id_mapping: ps.DataFrame,
    connected_components: ps.DataFrame,
) -> Dict[str, ps.DataFrame]:
    """
    Compute EC Core Entities connectivity metrics.

    Evaluates how well EC Core Entities (drugs and diseases) are integrated into
    the knowledge graph structure by analyzing their distribution across connected
    components.

    Mathematical definitions:
        N_EC = Total number of EC Core Entities
        S = Set of subgraphs (connected components)
        |S| = Number of subgraphs
        |S_i| = Size of subgraph i
        C_i = EC Core Entities in subgraph i
        |C_i| = Number of EC Core Entities in subgraph i
        N_LCC = Size of largest connected component (LCC)
        C_LCC = EC Core Entities in LCC

    Metrics:
        1. LCC Fraction = C_LCC / N_EC
           (Fraction of core entities in the main component)

        2. Weighted Connectivity Score = Σ(C_i/N_EC × S_i/N_LCC)
           (Weighted average considering both core entity count and component size)

    Args:
        nodes: Unified nodes DataFrame with 'id' and 'category' columns
        core_id_mapping: Core entity mapping with 'normalized_id' column
        connected_components: Component assignments with 'id' and 'component_id' columns

    Returns:
        Dictionary with keys:
        - "summary_metrics": Aggregate metrics by category (all_core, drugs, diseases)
        - "subgraph_details": Per-subgraph breakdown by category
    """
    logger.info("Computing EC Core Entities connectivity metrics...")

    # Join nodes with component assignments
    nodes_with_components = nodes.join(connected_components.select("id", "component_id"), on="id", how="inner")

    # Identify core entities by joining with core_id_mapping
    core_entities = nodes_with_components.join(
        core_id_mapping.select("normalized_id"), nodes_with_components.id == core_id_mapping.normalized_id, how="inner"
    ).select(nodes_with_components["*"])

    logger.info(f"Identified {core_entities.count():,} EC Core Entities in the graph")

    # Calculate component sizes (all nodes, not just core)
    component_sizes = nodes_with_components.groupBy("component_id").agg(F.count("*").alias("component_size"))

    # Get LCC size
    lcc_size = component_sizes.agg(F.max("component_size")).collect()[0][0]
    logger.info(f"Largest connected component has {lcc_size:,} nodes")

    # Calculate metrics for each category
    categories = [
        ("all_core", core_entities),
        ("drugs", core_entities.filter(F.col("category") == "biolink:Drug")),
        ("diseases", core_entities.filter(F.col("category") == "biolink:Disease")),
    ]

    summary_data = []
    detail_data = []

    for category_name, category_df in categories:
        logger.info(f"\nCalculating metrics for category: {category_name}")

        # Count total core entities in this category
        n_ec = category_df.count()
        logger.info(f"  Total {category_name} entities: {n_ec:,}")

        if n_ec == 0:
            logger.warning(f"  No entities found for {category_name}, skipping")
            continue

        # Count core entities per component
        core_per_component = category_df.groupBy("component_id").agg(F.count("*").alias("core_entity_count"))

        # Join with component sizes
        component_stats = component_sizes.join(core_per_component, on="component_id", how="left").fillna(
            0, subset=["core_entity_count"]
        )

        # Calculate additional metrics
        component_stats = (
            component_stats.withColumn("core_entity_fraction", F.col("core_entity_count") / F.lit(n_ec))
            .withColumn("size_relative_to_lcc", F.col("component_size") / F.lit(lcc_size))
            .withColumn("is_lcc", F.col("component_size") == F.lit(lcc_size))
        )

        # Get LCC stats
        lcc_stats = component_stats.filter(F.col("is_lcc")).select("core_entity_count").first()
        c_lcc = lcc_stats["core_entity_count"] if lcc_stats else 0

        # Calculate LCC Fraction
        lcc_fraction = c_lcc / n_ec if n_ec > 0 else 0.0

        # Calculate Weighted Connectivity Score
        # Σ(C_i/N_EC × S_i/N_LCC)
        weighted_score_df = component_stats.withColumn(
            "weighted_contribution",
            (F.col("core_entity_count") / F.lit(n_ec)) * (F.col("component_size") / F.lit(lcc_size)),
        )
        weighted_score = weighted_score_df.agg(F.sum("weighted_contribution")).collect()[0][0] or 0.0

        # Count subgraphs
        num_subgraphs = component_stats.count()

        logger.info(f"  Number of subgraphs: {num_subgraphs:,}")
        logger.info(f"  Core entities in LCC: {c_lcc:,}")
        logger.info(f"  LCC Fraction: {lcc_fraction:.4f}")
        logger.info(f"  Weighted Connectivity Score: {weighted_score:.4f}")

        # Add to summary
        summary_data.append(
            {
                "category": category_name,
                "total_core_entities": n_ec,
                "num_subgraphs": num_subgraphs,
                "lcc_size": lcc_size,
                "core_entities_in_lcc": c_lcc,
                "lcc_fraction": round(lcc_fraction, 6),
                "weighted_connectivity_score": round(weighted_score, 6),
            }
        )

        # Add to details (with category column)
        category_details = component_stats.select(
            F.lit(category_name).alias("category"),
            "component_id",
            "component_size",
            "core_entity_count",
            "core_entity_fraction",
            "size_relative_to_lcc",
            "is_lcc",
        ).orderBy(F.desc("component_size"))

        detail_data.append(category_details)

    # Create summary DataFrame
    spark = nodes.sparkSession
    summary_metrics = spark.createDataFrame(summary_data)

    # Union all detail DataFrames
    subgraph_details = detail_data[0]
    for df in detail_data[1:]:
        subgraph_details = subgraph_details.union(df)

    logger.info("\n" + "=" * 80)
    logger.info("EC Core Connectivity Metrics Summary:")
    logger.info("=" * 80)
    summary_metrics.show(truncate=False)

    return {
        "summary_metrics": summary_metrics,
        "subgraph_details": subgraph_details,
    }
