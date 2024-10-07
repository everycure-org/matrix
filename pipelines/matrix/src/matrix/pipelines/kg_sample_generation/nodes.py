import pandas as pd
from pyspark.sql import DataFrame
from typing import Tuple

def get_random_selection_of_edges(
    nodes: DataFrame,
    edges: DataFrame
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
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.05, seed=123)
    edges_node_ids_df = edges_sample_df.select('subject').union(edges_sample_df.select('object')).distinct()
    # Now we need to select the nodes that have been included in the selection of edges
    cond = (nodes.id == edges_node_ids_df.subject)
    nodes_sample_df = nodes.join(edges_node_ids_df, how='inner', on=cond).select(nodes.columns)
    return edges_sample_df, nodes_sample_df



