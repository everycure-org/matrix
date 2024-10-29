import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from typing import Tuple
import pyspark.sql.functions as F

from matrix.pipelines.modelling.nodes import create_int_pairs

def get_random_selection_from_rtx(
    nodes : DataFrame,
    edges : DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """Function that accepts the whole Edges and Nodes
       and makes a random selection of the Edges.
       The idea is to have a dataset that resembles the
       original KG data but which is much smaller so
       it can be ran on a local environment

    Args:
        nodes: Dataframe with raw nodes
        edges: Dataframe with raw edges
    """

    # Make the sample
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.0000005, seed=123)
    
    # Now we need to select the nodes. First, make a list with all the IDs
    edges_node_ids_df = (
        edges_sample_df.select('subject').withColumnRenamed("subject", "id")
        .unionByName(
            edges_sample_df.select('object').withColumnRenamed("object", "id")
        )
    ).distinct()

    nodes_sample_df = nodes.join(edges_node_ids_df, how='inner', on="id").select(nodes.columns)

    return {
        'nodes': nodes_sample_df,
        'edges': edges_sample_df
    }


def get_random_selection_from_robokop(
    nodes : DataFrame,
    edges : DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """Function that accepts the whole Edges and Nodes
       and makes a random selection of the Edges.
       The idea is to have a dataset that resembles the
       original KG data but which is much smaller so
       it can be ran on a local environment

    Args:
        nodes: Dataframe with raw nodes
        edges: Dataframe with raw edges
    """

    # Make the sample
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.00005, seed=123)
    
    # Now we need to select the nodes. First, make a list with all the IDs
    edges_node_ids_df = (
        edges_sample_df.select('subject:START_ID').withColumnRenamed("subject:START_ID", "id:ID")
        .unionByName(
            edges_sample_df.select('object:END_ID').withColumnRenamed("object:END_ID", "id:ID")
        )
    ).distinct()

    nodes_sample_df = nodes.join(edges_node_ids_df, how='inner', on="id:ID").select(nodes.columns)

    return {
        'nodes': nodes_sample_df,
        'edges': edges_sample_df
    }


def get_random_selection_of_ground_truths(
    ground_truth_either_df : pd.DataFrame
) -> pd.DataFrame:
    return ground_truth_either_df.sample(frac=0.1, replace=False, random_state=123)


