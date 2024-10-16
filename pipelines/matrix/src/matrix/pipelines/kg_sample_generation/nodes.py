import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from typing import Tuple
import pyspark.sql.functions as F


def get_random_selection_from_rtx(
    nodes : DataFrame,
    edges : DataFrame,
    known_pairs : pd.DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """Function that accepts the whole Edges and Nodes
       and makes a random selection of the Edges.
       The idea is to have a dataset that resembles the
       original KG data but which is much smaller so
       it can be ran on a local environment

    Args:
        nodes: Dataframe with raw nodes
        edges: Dataframe with raw edges
        known_pairs: Ground Truths
    """
    # First get the edges from the ground_truths as they are needed in the modelling pipeline: 
    # in make_splits() specifically
    spark_session = SparkSession.builder.getOrCreate()
    ground_truths_spark_df = spark_session.createDataFrame(known_pairs[['source', 'target']])
    # Joe 15/10/24: GTs edges are not necessarily represented in the KG. So 
    # first get the edges that are present, and then add the GT nodes to the sample
    cond = ((edges.subject == ground_truths_spark_df.source) & (edges.object == ground_truths_spark_df.target)) 
    ground_truths_edges = edges.join(ground_truths_spark_df, how='inner', on=cond).select(edges.columns)
    # Now take a sample from the whole KG edges
    edges_sample_df = edges.sample(withReplacement=False, fraction=0.005, seed=123)
    # Merge the ground truths
    edges_sample_df = edges_sample_df.union(ground_truths_edges).distinct()
    # Now we need to select the nodes. First, make a list with all the IDs
    edges_node_ids_df = edges_sample_df.select('subject').union(edges_sample_df.select('object')).distinct()
    # --- Joe 15/10/24: As a work-around, add the node IDs from the GT that were not represented in the KG:
    edges_node_ids_df = edges_node_ids_df.union(ground_truths_spark_df.select('source')).distinct()
    edges_node_ids_df = edges_node_ids_df.union(ground_truths_spark_df.select('target')).distinct()
    # --- end work-around
    # Now we need to select the nodes that have been included in the selection of edges
    cond = (nodes.id == edges_node_ids_df.subject)
    nodes_sample_df = nodes.join(edges_node_ids_df, how='inner', on=cond).select(nodes.columns)
    return {
        'nodes': nodes_sample_df.withColumn("kg_source", F.lit("rtx_kg2")),
        'edges': edges_sample_df.withColumn("kg_source", F.lit("rtx_kg2"))
    }



