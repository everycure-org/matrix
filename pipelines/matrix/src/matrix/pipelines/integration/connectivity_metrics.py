import gc
import logging
import time
from collections import Counter
from functools import reduce
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
    logger.info(f"Simplifying graph: {edges.count():,} edges before deduplication")

    # Filter self-loops and keep only unique (subject, object) pairs
    simplified = edges.select("subject", "object").filter(F.col("subject") != F.col("object")).distinct()

    reduction_pct = 100 * (1 - simplified.count() / edges.count()) if edges.count() > 0 else 0

    logger.info(
        f"Graph simplified: {simplified.count():,} unique edges ({reduction_pct:.1f}% reduction from {edges.count():,})"
    )

    return simplified


def compute_connected_components_graphframes(nodes: ps.DataFrame, edges: ps.DataFrame) -> Dict[str, ps.DataFrame]:
    """
    Compute connected components using GraphFrames (Spark-native).

    GraphFrames is a graph processing library built on Apache Spark DataFrames.
    It leverages Spark's distributed computing for scalability on large graphs.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        Dictionary with keys:
        - "node_assignments": DataFrame with (id, component_id) - node-to-component mapping
        - "component_stats": DataFrame with (component_id, component_size, num_components) - statistics
    """
    from graphframes import GraphFrame

    logger.info("Computing connected components using GraphFrames...")

    gf_nodes = nodes.select(F.col("id"))

    # Prepare edges for GraphFrame (needs 'src' and 'dst' columns)
    gf_edges = edges.select(F.col("subject").alias("src"), F.col("object").alias("dst"))

    graph = GraphFrame(gf_nodes, gf_edges)

    spark = nodes.sparkSession
    checkpoint_dir = "/data/checkpoints/graphframes"
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    components = graph.connectedComponents()

    node_assignments = components.select(F.col("id"), F.col("component").alias("component_id"))

    num_components = components.select("component").distinct().count()
    component_stats = (
        components.groupBy("component")
        .agg(F.count("*").alias("component_size"))
        .withColumnRenamed("component", "component_id")
        .withColumn("num_components", F.lit(num_components))
        .orderBy(F.desc("component_size"))
    )

    logger.info(
        f"Found {num_components:,} connected components using GraphFrames. "
        f"Largest component has {component_stats.first()['component_size']:,} nodes."
    )

    return {
        "node_assignments": node_assignments,
        "component_stats": component_stats,
    }


def compute_connected_components_grape(nodes: ps.DataFrame, edges: ps.DataFrame) -> Dict[str, ps.DataFrame]:
    """
    Compute connected components using grape (Rust-based, ultra-fast).

    grape/ensmallen is a Rust-based graph processing library designed for
    billion-scale graphs with ultra-fast performance.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        Dictionary with keys:
        - "node_assignments": DataFrame with (id, component_id) - node-to-component mapping
        - "component_stats": DataFrame with (component_id, component_size, num_components) - statistics
    """
    from ensmallen import GraphBuilder

    logger.info("Computing connected components using grape...")

    # Collect to driver for in-memory processing (grape requires non-distributed data)
    builder = GraphBuilder()
    builder.set_directed(False)
    builder.set_name("Connectivity_Graph")

    nodes_list = [row.id for row in nodes.select("id").collect()]
    for node_id in nodes_list:
        builder.add_node(name=node_id)

    edges_list = edges.select("subject", "object").collect()
    for row in edges_list:
        builder.add_edge(src=row.subject, dst=row.object)

    logger.info(f"Building grape graph with {len(nodes_list):,} nodes and {len(edges_list):,} edges...")
    graph = builder.build()

    logger.info("Computing connected components with grape...")
    component_per_node, num_components, min_size, max_size = graph.get_connected_components()

    component_sizes = Counter(component_per_node)

    # Convert results back to Spark DataFrames for pipeline compatibility
    # Use RDD with explicit partitioning to avoid large task closures
    spark = nodes.sparkSession
    node_assignment_data = [
        {"id": node_id, "component_id": int(comp_id)} for node_id, comp_id in zip(nodes_list, component_per_node)
    ]
    node_assignments = spark.createDataFrame(spark.sparkContext.parallelize(node_assignment_data, numSlices=500))

    component_data = [
        {"component_id": int(comp_id), "component_size": int(size), "num_components": int(num_components)}
        for comp_id, size in component_sizes.items()
    ]
    component_stats = spark.createDataFrame(spark.sparkContext.parallelize(component_data, numSlices=200)).orderBy(
        F.desc("component_size")
    )

    logger.info(
        f"Found {num_components:,} connected components using grape. "
        f"Largest component has {max(component_sizes.values()):,} nodes."
    )

    return {
        "node_assignments": node_assignments,
        "component_stats": component_stats,
    }


def compute_connected_components_rustworkx(nodes: ps.DataFrame, edges: ps.DataFrame) -> Dict[str, ps.DataFrame]:
    """
    Compute connected components using rustworkx (Rust-based, high-performance).

    rustworkx is a high-performance graph library written in Rust with Python bindings,
    designed as a faster alternative to NetworkX.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        Dictionary with keys:
        - "node_assignments": DataFrame with (id, component_id) - node-to-component mapping
        - "component_stats": DataFrame with (component_id, component_size, num_components) - statistics
    """
    import rustworkx as rx

    logger.info("Computing connected components using rustworkx...")

    # Collect to driver for in-memory processing (rustworkx requires non-distributed data)
    graph = rx.PyGraph()

    nodes_list = [row.id for row in nodes.select("id").collect()]
    node_indices = graph.add_nodes_from(nodes_list)

    id_to_index = {node_id: idx for node_id, idx in zip(nodes_list, node_indices)}
    index_to_id = {idx: node_id for node_id, idx in id_to_index.items()}

    edges_list = edges.select("subject", "object").collect()

    edge_tuples = [
        (id_to_index[row.subject], id_to_index[row.object], None)
        for row in edges_list
        if row.subject in id_to_index and row.object in id_to_index
    ]

    graph.add_edges_from(edge_tuples)

    logger.info(f"Built rustworkx graph with {graph.num_nodes():,} nodes and {graph.num_edges():,} edges")

    logger.info("Computing connected components with rustworkx...")
    components = rx.connected_components(graph)
    num_components = len(components)

    # Convert results back to Spark DataFrames for pipeline compatibility
    # Use RDD with explicit partitioning to avoid large task closures
    spark = nodes.sparkSession
    node_assignment_data = [
        {"id": index_to_id[node_idx], "component_id": component_id}
        for component_id, node_indices_set in enumerate(components)
        for node_idx in node_indices_set
    ]
    node_assignments = spark.createDataFrame(spark.sparkContext.parallelize(node_assignment_data, numSlices=500))

    component_stats = spark.createDataFrame(
        spark.sparkContext.parallelize(
            [
                {"component_id": i, "component_size": len(c), "num_components": num_components}
                for i, c in enumerate(components)
            ],
            numSlices=200,
        )
    ).orderBy(F.desc("component_size"))

    logger.info(
        f"Found {num_components:,} connected components using rustworkx. "
        f"Largest component has {component_stats.first()['component_size']:,} nodes."
    )

    return {
        "node_assignments": node_assignments,
        "component_stats": component_stats,
    }


def compute_connected_components(nodes: ps.DataFrame, edges: ps.DataFrame, algorithm: str) -> Dict[str, ps.DataFrame]:
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
        Dictionary with keys:
        - "node_assignments": DataFrame with (id, component_id) - node-to-component mapping
        - "component_stats": DataFrame with (component_id, component_size, num_components) - statistics

    Raises:
        ValueError: If algorithm is not one of the supported options
    """
    valid_algorithms = ["graphframes", "grape", "rustworkx", "benchmark"]
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of: {', '.join(valid_algorithms)}")

    logger.info(f"Computing connected components with algorithm: {algorithm}")

    simplified_edges = _simplify_graph_for_connectivity(edges)

    if algorithm == "benchmark":
        return _run_benchmark(nodes, simplified_edges)
    else:
        if algorithm == "graphframes":
            return compute_connected_components_graphframes(nodes, simplified_edges)
        elif algorithm == "grape":
            return compute_connected_components_grape(nodes, simplified_edges)
        elif algorithm == "rustworkx":
            return compute_connected_components_rustworkx(nodes, simplified_edges)


def _run_benchmark(nodes: ps.DataFrame, edges: ps.DataFrame) -> Dict[str, ps.DataFrame]:
    """
    Run all three connected components algorithms and compare performance.

    Benchmark results are logged but not saved. This is intended for testing
    and algorithm selection purposes only.

    Args:
        nodes: DataFrame with 'id' column
        edges: DataFrame with 'subject' and 'object' columns (should be simplified)

    Returns:
        Dictionary with node assignments and component statistics from graphframes algorithm
    """

    benchmark_results = []
    graphframes_output = None

    algorithms = [
        ("graphframes", compute_connected_components_graphframes),
        ("grape", compute_connected_components_grape),
        ("rustworkx", compute_connected_components_rustworkx),
    ]

    for algo_name, algo_func in algorithms:
        logger.info(f"Running {algo_name.upper()}...")
        start_time = time.time()
        results = algo_func(nodes, edges)
        execution_time = time.time() - start_time

        first_row = results["component_stats"].first()
        benchmark_results.append(
            {
                "algorithm": algo_name,
                "execution_time_seconds": round(execution_time, 2),
                "num_components": first_row["num_components"],
                "largest_component_size": first_row["component_size"],
            }
        )

        # Only keep graphframes results for return value, discard others to save memory
        if algo_name == "graphframes":
            graphframes_output = results

        logger.info(f"{algo_name.upper()} completed in {execution_time:.2f} seconds")
        del results  # Explicitly delete to help garbage collection
        gc.collect()

    logger.info("Benchmark Results (sorted by execution time):")
    for result in sorted(benchmark_results, key=lambda x: x["execution_time_seconds"]):
        logger.info(
            f"{result['algorithm']:12s}: {result['execution_time_seconds']:8.2f}s "
            f"| Components: {result['num_components']:,} "
            f"| Largest: {result['largest_component_size']:,}"
        )

    return graphframes_output


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

    nodes_with_components = nodes.join(connected_components.select("id", "component_id"), on="id", how="inner")

    core_entities = nodes_with_components.join(
        core_id_mapping.select("normalized_id", "category"),
        nodes_with_components.id == core_id_mapping.normalized_id,
        how="inner",
    ).select(nodes_with_components["*"], core_id_mapping["category"].alias("core_category"))

    component_sizes = nodes_with_components.groupBy("component_id").agg(F.count("*").alias("component_size"))

    # Get LCC size
    lcc_size = component_sizes.agg(F.max("component_size")).collect()[0][0]
    logger.info(f"Largest connected component has {lcc_size:,} nodes")

    # Calculate metrics for each category
    categories = [
        ("all_core", core_entities),
        ("drugs", core_entities.filter(F.col("core_category") == "biolink:Drug")),
        ("diseases", core_entities.filter(F.col("core_category") == "biolink:Disease")),
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

        # Count core entities per component and join with component sizes
        core_per_component = category_df.groupBy("component_id").agg(F.count("*").alias("core_entity_count"))

        component_stats = (
            component_sizes.join(core_per_component, on="component_id", how="left")
            .fillna(0, subset=["core_entity_count"])
            .withColumn("core_entity_fraction", F.col("core_entity_count") / F.lit(n_ec))
            .withColumn("size_relative_to_lcc", F.col("component_size") / F.lit(lcc_size))
            .withColumn("is_lcc", F.col("component_size") == F.lit(lcc_size))
        )

        # Get LCC stats
        lcc_stats = component_stats.filter(F.col("is_lcc")).select("core_entity_count").first()
        c_lcc = lcc_stats["core_entity_count"] if lcc_stats else 0

        lcc_fraction = c_lcc / n_ec if n_ec > 0 else 0.0

        # Calculate Weighted Connectivity Score: Σ(C_i/N_EC × S_i/N_LCC)
        weighted_score = (
            component_stats.agg(
                F.sum((F.col("core_entity_count") / F.lit(n_ec)) * (F.col("component_size") / F.lit(lcc_size)))
            ).collect()[0][0]
            or 0.0
        )

        num_subgraphs = component_stats.count()

        logger.info(f"  Number of subgraphs: {num_subgraphs:,}")
        logger.info(f"  Core entities in LCC: {c_lcc:,}")
        logger.info(f"  LCC Fraction: {lcc_fraction:.4f}")
        logger.info(f"  Weighted Connectivity Score: {weighted_score:.4f}")

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

    summary_metrics = nodes.sparkSession.createDataFrame(summary_data)
    subgraph_details = reduce(lambda df1, df2: df1.union(df2), detail_data)

    logger.info("EC Core Connectivity Metrics Summary:")
    summary_metrics.show(truncate=False)

    return {"summary_metrics": summary_metrics, "subgraph_details": subgraph_details}
