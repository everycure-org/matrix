import pandas as pd
from pyspark.sql import DataFrame
from typing import Tuple


def get_random_sample_from_unified_files(
    nodes: DataFrame,
    edges: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    '''
    Function accepts unified nodes and edges,
    and returns a random selection of both.

    Args:
        nodes_sdf: Spark Dataframe with unified nodes
        edges_sdf: Spark Dataframe with unified edges
    '''
    # Take the sample
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.00005, seed=123)
    # Now we need to select the nodes from the selected edges.
    # First, make a list with all the IDs
    edges_node_ids_df = (
        edges_sample_df.select('subject').withColumnRenamed("subject", "id")
        .unionByName(
            edges_sample_df.select('object').withColumnRenamed("object", "id")
        )
    ).distinct()
    nodes_sample_df = nodes.join(
        edges_node_ids_df, how='inner', on="id"
    ).select(nodes.columns)
    return {
        'nodes': nodes_sample_df,
        'edges': edges_sample_df
    }


def get_random_selection_of_ground_truths(
    ground_truth_either_df : pd.DataFrame
) -> pd.DataFrame:
    return ground_truth_either_df.sample(frac=0.1, replace=False, random_state=123)


