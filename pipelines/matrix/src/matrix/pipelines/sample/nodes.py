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
    def _union_cols(gt: DataFrame) -> DataFrame:
        """Helper function to enable unioning of grount truth nodes."""
        return gt.withColumn("id", F.col("source")).unionByName(gt.withColumn("id", F.col("target"))).select("id")

    def _make_joinable(df: DataFrame, column: str) -> DataFrame:
        """Helper function to simplify joining nodes on edges df."""
        return df.withColumn(column, F.col("id")).select(column)

    # Sample GT
    gt_positives_sample = gt_positives.sample(frac=0.05, replace=False, random_state=123)
    gt_negatives_sample = gt_negatives.sample(frac=0.05, replace=False, random_state=123)

    # Sample additional nodes
    nodes_sample = nodes.sample(frac=0.01, replace=False, random_state=123).select("id")

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
        .sample(frac=0.05, replace=False, random_state=123)
    )

    return {
        "gt_positives": gt_positives_sample,
        "gt_negatives": gt_negatives_sample,
        "nodes": node_selection,
        "edges": edge_selection,
    }
