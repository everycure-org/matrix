import pandas as pd
from pyspark.sql import DataFrame
from typing import Tuple
import pyspark.sql.functions as F

from matrix.pipelines.modelling.nodes import create_int_pairs

import logging
logger = logging.getLogger(__name__)

def get_random_selection_from_rtx(
    nodes : DataFrame,
    edges : DataFrame,
    raw_tp: pd.DataFrame,
    raw_tn: pd.DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """Function that accepts the whole Edges and Nodes
       and makes a random selection of the Edges.
       The idea is to have a dataset that resembles the
       original KG data but which is much smaller so
       it can be ran on a local environment

    Args:
        nodes: Dataframe with raw nodes
        edges: Dataframe with raw edges
        raw_tp: Raw ground truth positive data.
        raw_tn: Raw ground truth negative data.
    """
    # First get the edges from the ground_truths as they are needed in the modelling pipeline
    ground_truths_df = create_int_pairs(raw_tp,raw_tn)
    ground_truths_edges = edges.filter((edges.subject == ground_truths_df.source) | (edges.object == ground_truths_df.target))
    # Now take a sample from the whole KG edges
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.05, seed=123)
    # Now merge the ground truths
    edges_sample_df = edges_sample_df.union(ground_truths_edges).distinct()
    # Now we need to select the nodes. First, make a list with all the IDs
    edges_node_ids_df = edges_sample_df.select('subject').union(edges_sample_df.select('object')).distinct()
    # Now we need to select the nodes that have been included in the selection of edges
    cond = (nodes.id == edges_node_ids_df.subject)
    nodes_sample_df = nodes.join(edges_node_ids_df, how='inner', on=cond).select(nodes.columns)
    return {
        'nodes': nodes_sample_df.withColumn("kg_source", F.lit("rtx_kg2")),
        'edges': edges_sample_df.withColumn("kg_source", F.lit("rtx_kg2"))
    }



