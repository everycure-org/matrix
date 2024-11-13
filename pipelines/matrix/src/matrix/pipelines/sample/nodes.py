import pandas as pd
from pyspark.sql import DataFrame
from typing import Tuple

import pyspark.sql.functions as F


def sample(
    gt_positives: pd.DataFrame,
    gt_negatives: pd.DataFrame,
    nodes: DataFrame,
    edges: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """Function to sample datasets.

    Sampling is currently performed by sampling ground truth data, and grabbing
    a subset of nodes N that contains the ground truth subset. Edges are thereafter
    restricted to edges that connect nodes in N, and a fraction of those edges is sampled.

    Args:
        gt_positives: gt positives
        gt_negatives: gt negatives
        nodes: nodes df
        edges: edges df
    Returns:
        Filtered artifacts
    """

    def _union_cols(gt: DataFrame) -> DataFrame:
        """Helper function to enable unioning of grount truth nodes."""
        return gt.withColumn("id", F.col("source")).unionByName(gt.withColumn("id", F.col("target"))).select("id")

    def _make_joinable(df: DataFrame, column: str) -> DataFrame:
        """Helper function to simplify joining nodes on edges df."""
        return df.withColumn(column, F.col("id")).select(column)

    # Sample GT
    gt_positives_sample = gt_positives.sample(fraction=0.05, withReplacement=False, seed=123)
    gt_negatives_sample = gt_negatives.sample(fraction=0.05, withReplacement=False, seed=123)

    # Sample additional nodes
    nodes_sample = nodes.sample(fraction=0.1, withReplacement=False, seed=123).select("id")

    # Construct node selection
    node_selection = (
        nodes_sample.unionByName(_union_cols(gt_negatives_sample))
        .unionByName(_union_cols(gt_positives_sample))
        .distinct()
        .join(nodes, on="id", how="left")
    )

    # Construct edge selection
    edge_selection = (
        edges.join(_make_joinable(node_selection, column="subject"), on="subject")
        .join(_make_joinable(node_selection, column="object"), on="object")
        .sample(fraction=0.05, withReplacement=False, seed=123)
    )

    return {
        "gt_positives": gt_positives_sample,
        "gt_negatives": gt_negatives_sample,
        "nodes": node_selection,
        "edges": edge_selection,
    }


def reduce_embeddings(embeddings: pd.DataFrame, nodes: pd.DataFrame):
    """Function to restrict embeddings output to relevant nodes.

    Args:
        embeddings: embeddings
        nodes: nodes
    Returns:
        filtered embeddings
    """
    return embeddings.alias("embeddings").join(nodes, on="id").select("embeddings.*")
