"""
This is a boilerplate pipeline 'sampling'
generated using Kedro 0.19.6
"""
# NOTE: This file was partially generated using AI assistance. Delete this line when it was properly reviewed by a human.

import logging
from typing import Any, Dict, Tuple

import networkx as nx
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StringType, StructField, StructType

logger = logging.getLogger(__name__)


def select_sample_ids(gt: DataFrame, drugs: DataFrame, diseases: DataFrame, params: Dict[str, Any]) -> DataFrame:
    """Select a sample of IDs from ground truth, drugs, and diseases.

    Args:
        gt: Ground truth DataFrame with 'source' column
        drugs: Drugs DataFrame with 'curie' column
        diseases: Diseases DataFrame with 'curie' column
        params: Parameters dictionary containing 'max_count' for sampling

    Returns:
        DataFrame: Selected sample IDs
    """

    # TODO still needs to be improved using object injection
    count = params.get("max_count", 150) // 4
    ids_always_keep = params.get("always_keep_ids", [])

    # Get the spark session from one of the input DataFrames
    spark = gt.sparkSession

    # Sample from each source
    gt_sampled = gt.orderBy(gt.source).select(gt.source).distinct().limit(count)
    drugs_sampled = drugs.orderBy(drugs.curie).select(drugs.curie).distinct().limit(count)
    diseases_sampled = diseases.orderBy(diseases.curie).select(diseases.curie).distinct().limit(count)

    # Optional always-keep IDs (can be configured via params)
    ids_always_keep_df = spark.createDataFrame(
        [[id] for id in ids_always_keep], schema=StructType([StructField("id", StringType(), True)])
    )

    # Combine all samples
    sample = None
    for df in [gt_sampled, drugs_sampled, diseases_sampled, ids_always_keep_df]:
        df_exploded = df.select(f.explode(f.array(*[f.col(c) for c in df.columns])).alias("id"))
        sample = df_exploded if sample is None else sample.union(df_exploded)

    return sample.distinct()


def sample_kg_from_ids(
    node_ids: DataFrame, nodes: DataFrame, edges: DataFrame, params: Dict[str, Any]
) -> Tuple[DataFrame, DataFrame]:
    """Sample a knowledge graph based on selected node IDs.

    Args:
        sampled_nodes: DataFrame containing selected node IDs
        nodes: Full nodes DataFrame
        edges: Full edges DataFrame
        params: Parameters dictionary containing optional 'max_edges'

    Returns:
        Tuple[DataFrame, DataFrame]: Sampled nodes and edges DataFrames
    """
    # Get nodes that match our sample
    nodes_sampled, edges_sampled = get_sampled_nodes_and_neighors(nodes, edges, node_ids, params.get("max_edges"))

    subgraph_nodes, subgraph_edges = get_subgraph_via_networkX(nodes_sampled, edges_sampled)

    # Join back with original data to get full node/edge information
    final_nodes = nodes.join(subgraph_nodes, on="id", how="inner")
    final_edges = edges_sampled.join(subgraph_edges, on=["subject", "object"], how="inner")

    # get total counts
    logger.info(f"Total nodes sampled: {final_nodes.count()}")
    logger.info(f"Total edges sampled: {final_edges.count()}")

    # few asserts to make sure we're doing the right thing
    # are we only returning edges that are present in the nodes?
    assert (
        final_nodes.select("id")
        .distinct()
        .join(final_edges.select(f.explode(f.array("subject", "object")).alias("id")).distinct(), on="id", how="inner")
        .count()
        == final_nodes.count()
    )

    return final_nodes, final_edges


def get_subgraph_via_networkX(nodes_sampled: DataFrame, edges_sampled: DataFrame) -> nx.Graph:
    # Convert to NetworkX for component analysis
    nodes_pd = nodes_sampled.select("id", "name").toPandas()
    edges_pd = edges_sampled.select("subject", "predicate", "object").toPandas()

    G = nx.DiGraph()
    for _, row in nodes_pd.iterrows():
        G.add_node(row["id"], name=row["name"])
    for _, row in edges_pd.iterrows():
        G.add_edge(row["subject"], row["object"], type=row["predicate"])

    # Get largest connected component
    components = list(nx.connected_components(G.to_undirected()))
    logger.info(f"Number of components in graph: {len(components)}")
    largest_component = max(components, key=len)
    largest_subgraph = G.subgraph(largest_component).copy()

    # Log graph statistics
    logger.info(f"Largest component nodes: {largest_subgraph.number_of_nodes()}")
    logger.info(f"Largest component edges: {largest_subgraph.number_of_edges()}")
    logger.info(f"Average degree: {sum(dict(largest_subgraph.degree()).values()) / largest_subgraph.number_of_nodes()}")

    # Convert back to Spark DataFrames
    spark = nodes_sampled.sparkSession
    subgraph_nodes = spark.createDataFrame(
        pd.DataFrame(largest_subgraph.nodes()), schema=StructType([StructField("id", StringType(), True)])
    )
    subgraph_edges = spark.createDataFrame(
        pd.DataFrame(largest_subgraph.edges()),
        schema=StructType([StructField("subject", StringType(), True), StructField("object", StringType(), True)]),
    )

    return subgraph_nodes, subgraph_edges


def get_sampled_nodes_and_neighors(
    nodes: DataFrame, edges: DataFrame, node_ids: DataFrame, max_edges: int
) -> Tuple[DataFrame, DataFrame]:
    nodes_sampled = nodes.join(node_ids, on="id", how="inner").persist()
    logger.info(f"Nodes found in KG: {nodes_sampled.count()}")

    # Get edges connected to our sampled nodes
    edges_sampled = edges.join(
        node_ids, on=(edges.subject == node_ids.id) | (edges.object == node_ids.id), how="inner"
    ).persist()
    logger.info(f"Edges selected from KG: {edges_sampled.count()}")

    # Optionally limit edges
    if max_edges:
        edges_sampled = edges_sampled.limit(max_edges)
        logger.info(f"Edges limited to: {edges_sampled.count()}")

    return nodes_sampled, edges_sampled
