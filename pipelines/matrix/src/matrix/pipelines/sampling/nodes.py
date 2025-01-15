import logging

import pyspark.sql as ps
from pyspark.sql import functions as f

logger = logging.getLogger(__name__)


def sample(
    kg_nodes: ps.DataFrame,
    kg_edges: ps.DataFrame,
    gt_p: ps.DataFrame,
    gt_n: ps.DataFrame,
    embeddings_nodes: ps.DataFrame,
    ground_truth_positive_sample_ratio: float,
    ground_truth_negative_sample_ratio: float,
    kg_nodes_sample_fraction: float,
    seed: int,
    **kwargs,
):
    """
    Sample the KG, GT, and embeddings nodes and edges.

    Args:
        kg_nodes: Knowledge graph nodes
        kg_edges: Knowledge graph edges
        gt_p: Ground truth positives
        gt_n: Ground truth negatives
        embeddings_nodes: Embeddings nodes
        gt_p_count: Number of ground truth positives to sample
        gt_n_count: Number of ground truth negatives to sample
    """
    logger.info(f"Received {kg_nodes.count()} KG nodes")
    logger.info(f"Received {kg_edges.count()} KG edges")
    logger.info(f"Received {gt_p.count()} ground truth positives")
    logger.info(f"Received {gt_n.count()} ground truth negatives")
    logger.info(f"Received {embeddings_nodes.count()} embeddings")

    # Select GT that are in the KG
    gt_tp_in_kg = (
        gt_p.join(kg_nodes.alias("source_nodes"), gt_p.source == f.col("source_nodes.id"))
        .join(kg_nodes.alias("target_nodes"), gt_p.target == f.col("target_nodes.id"))
        .select(gt_p["*"])
    )
    gt_tn_in_kg = (
        gt_n.join(kg_nodes.alias("source_nodes"), gt_n.source == f.col("source_nodes.id"))
        .join(kg_nodes.alias("target_nodes"), gt_n.target == f.col("target_nodes.id"))
        .select(gt_n["*"])
    )

    # Sample from GT
    sampled_gt_p = gt_tp_in_kg.sample(ground_truth_positive_sample_ratio, seed)
    sampled_gt_n = gt_tn_in_kg.sample(ground_truth_negative_sample_ratio, seed)

    # Select nodes attached to sampled GT
    all_sampled_gt = sampled_gt_p.union(sampled_gt_n)
    gt_node_ids = (
        all_sampled_gt.withColumn("ids", f.array("source", "target")).select(f.explode("ids").alias("id")).distinct()
    )

    # Sample from remaining nodes
    sampled_kg_node_ids = kg_nodes.select("id").sample(kg_nodes_sample_fraction, seed)

    # Drop duplicated node ids
    sampled_node_ids = sampled_kg_node_ids.union(gt_node_ids).distinct()

    # Join on nodes
    sampled_nodes = (
        kg_nodes.alias("kg_nodes")
        .join(sampled_node_ids, sampled_node_ids.id == kg_nodes.id, "inner")
        .select("kg_nodes.*")
    )

    # Join on embeddings
    sampled_embeddings_nodes = (
        embeddings_nodes.alias("embeddings_nodes")
        .join(sampled_node_ids, embeddings_nodes.id == sampled_node_ids.id, "inner")
        .select("embeddings_nodes.*")
    )

    logger.info(f"Sampled {sampled_gt_p.count()} ground truth positives")
    logger.info(f"Sampled {sampled_gt_n.count()} ground truth negatives")
    logger.info(f"Sampled {gt_node_ids.count()} sampled unique nodes from GT")
    logger.info(f"Sampled {sampled_kg_node_ids.count()} KG nodes")
    logger.info(f"Sampled {sampled_nodes.count()} sampled nodes in total")
    logger.info(f"Sampled {sampled_embeddings_nodes.count()} embeddings nodes")

    return (sampled_nodes, sampled_gt_p, sampled_gt_n, sampled_embeddings_nodes)
