import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pyspark.sql as ps
import pyspark.sql.functions as f

logger = logging.getLogger(__name__)


class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        kg_nodes: ps.DataFrame,
        kg_edges: ps.DataFrame,
        gt_p: ps.DataFrame,
        gt_n: ps.DataFrame,
        embeddings_nodes: ps.DataFrame,
    ) -> Tuple[ps.DataFrame, ps.DataFrame, ps.DataFrame, ps.DataFrame]:
        """
        Sample the KG, ground truth, and embeddings.

        Args:
            kg_nodes: KG nodes
            kg_edges: KG edges
            gt_p: Ground truth positives
            gt_n: Ground truth negatives
            embeddings_nodes: Embeddings nodes

        Returns:
            sampled_kg_nodes: Sampled KG nodes
            sampled_gt_p: Sampled ground truth positives
            sampled_gt_n: Sampled ground truth negatives
            sampled_embeddings_nodes: Sampled embeddings nodes
        """


class GroundTruthSampler(Sampler):
    """
    A sampler that will sample from ground truth positives and negatives pairs, and then from the rest of the graph.
    """

    def __init__(
        self,
        ground_truth_positive_sample_ratio: float,
        ground_truth_negative_sample_ratio: float,
        kg_nodes_sample_ratio: float,
        seed: int,
    ):
        self.ground_truth_positive_sample_ratio = ground_truth_positive_sample_ratio
        self.ground_truth_negative_sample_ratio = ground_truth_negative_sample_ratio
        self.kg_nodes_sample_ratio = kg_nodes_sample_ratio
        self.seed = seed

    def sample(
        self,
        kg_nodes: ps.DataFrame,
        kg_edges: ps.DataFrame,
        gt_p: ps.DataFrame,
        gt_n: ps.DataFrame,
        embeddings_nodes: ps.DataFrame,
    ) -> Tuple[ps.DataFrame, ps.DataFrame, ps.DataFrame, ps.DataFrame]:
        logger.info(f"Received {kg_nodes.count()} KG nodes")
        logger.info(f"Received {kg_edges.count()} KG edges")
        logger.info(f"Received {gt_p.count()} ground truth positives")
        logger.info(f"Received {gt_n.count()} ground truth negatives")
        logger.info(f"Received {embeddings_nodes.count()} embeddings")

        # ... select GT edges that are in the KG
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

        # ... sample from GT edges
        sampled_gt_p = gt_tp_in_kg.sample(self.ground_truth_positive_sample_ratio, self.seed)
        sampled_gt_n = gt_tn_in_kg.sample(self.ground_truth_negative_sample_ratio, self.seed)

        # ... select nodes attached to sampled GT edges
        all_sampled_gt = sampled_gt_p.union(sampled_gt_n)
        gt_node_ids = (
            all_sampled_gt.withColumn("ids", f.array("source", "target"))
            .select(f.explode("ids").alias("id"))
            .distinct()
        )

        # ... sample from all KG nodes
        sampled_kg_node_ids = kg_nodes.select("id").sample(self.kg_nodes_sample_ratio, self.seed)

        # ... aggregate nodes and drop duplicated node ids
        sampled_node_ids = sampled_kg_node_ids.union(gt_node_ids).distinct()

        # ... join on KG nodes
        sampled_nodes = (
            kg_nodes.alias("kg_nodes")
            .join(sampled_node_ids, sampled_node_ids.id == kg_nodes.id, "inner")
            .select("kg_nodes.*")
        )

        # ... join on embeddings nodes
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

        return sampled_nodes, sampled_gt_p, sampled_gt_n, sampled_embeddings_nodes
