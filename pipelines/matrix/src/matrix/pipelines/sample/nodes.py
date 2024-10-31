import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from typing import Tuple

from matrix.pipelines.modelling.nodes import create_int_pairs

def get_random_selection_of_ground_truths(
    gt_positives : pd.DataFrame,
    gt_negatives : pd.DataFrame,
) -> pd.DataFrame:
    gt_positives_sample = gt_positives.sample(frac=0.05, replace=False, random_state=123)
    gt_negatives_sample = gt_negatives.sample(frac=0.05, replace=False, random_state=123)
    known_pairs = create_int_pairs(gt_positives_sample,gt_negatives_sample)
    return {
        'gt_positives': gt_positives_sample,
        'gt_negatives': gt_negatives_sample,
        'known_pairs': known_pairs
    }


def get_random_sample_from_unified_files(
    nodes: DataFrame,
    edges: DataFrame,
    known_pairs: pd.DataFrame
) -> Tuple[DataFrame, DataFrame]:
    '''
    Function accepts unified nodes and edges,
    and returns a random selection for both,
    and includes the Ground Truths in the nodes
    (and edges when present) of the sample.

    Args:
        nodes: Spark Dataframe with unified nodes
        edges: Spark Dataframe with unified edges
        known_pairs: Spark Dataframe with sampled Known Pairs
    '''
    # Convert the known_pairs Pandas DataFrame to Spark DataFrame
    spark_session = SparkSession.builder.getOrCreate()
    known_pairs_sdf = spark_session.createDataFrame(known_pairs[['source', 'target']])
    ### NOTE: the GT edges might not be present in the KG edges,
    #         so the following condition won't satisfy most GT edges, but we need
    #         to add them anyways to the selected nodes so they will have a
    #         corresponding calculated embedding
    cond = ((edges.subject == known_pairs_sdf.source) & (edges.object == known_pairs_sdf.target))
    known_pairs_edges = edges.join(known_pairs_sdf, how='inner', on=cond).select(edges.columns)
    # Take the sample from the whole unified KG
    edges_sample_sdf = edges.sample(withReplacement=False, fraction=0.00005, seed=123)
    # Now merge the ground truths
    edges_sample_sdf = edges_sample_sdf.union(known_pairs_edges).distinct()
    # Now we need to select the nodes from the selected edges.
    # First, make a list with all the node IDs from the edges
    edges_node_ids_sdf = (
        edges_sample_sdf.select('subject').withColumnRenamed("subject", "id")
        .unionByName(
            edges_sample_sdf.select('object').withColumnRenamed("object", "id")
        )
    ).dropDuplicates(['id'])
    nodes_sample_sdf = nodes.join(
        edges_node_ids_sdf, how='inner', on="id"
    ).select(nodes.columns)
    # Now add the nodes that are part of the GT but were not present
    # in the 'edges' of the KG
    missing_source_nodes = nodes.join(
        known_pairs_sdf.select('source').withColumnRenamed("source", "id"),
        how='inner', on='id'
    )
    missing_target_nodes = nodes.join(
        known_pairs_sdf.select('target').withColumnRenamed("target", "id"),
        how='inner', on='id'
    )
    # We need both the `id` and `category` for ALL nodes
    # (this includes the Ground Truth nodes) -- since:
    #  modelling.nodes.prefilter_nodes() line 66
    # will filter out all nodes that don't have `category`
    nodes_sample_sdf = nodes_sample_sdf.union(missing_source_nodes)
    nodes_sample_sdf = nodes_sample_sdf.union(missing_target_nodes)
    nodes_sample_sdf = nodes_sample_sdf.dropDuplicates(['id'])
    return {
        'nodes': nodes_sample_sdf,
        'edges': edges_sample_sdf
    }




