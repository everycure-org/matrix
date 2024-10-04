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
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.1, seed=123)
    edges_subcols_df = edges_sample_df.select('start_id', 'end_id')
    # Now we need to select the nodes that have been included in the selection of edges
    cond = (nodes.id == edges_subcols_df.start_id) | (nodes.id == edges_subcols_df.end_id)
    nodes_sample_df = nodes.join(edges_subcols_df, how='inner', on=cond).select(nodes.columns)
    return edges_sample_df, nodes_sample_df



