import logging
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple

import pyspark.sql as ps
import pyspark.sql.functions as f

logger = logging.getLogger(__name__)


class SamplingResult(NamedTuple):
    knowledge_graph_nodes: ps.DataFrame
    knowledge_graph_edges: ps.DataFrame


class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        knowledge_graph_nodes: ps.DataFrame,
        knowledge_graph_edges: ps.DataFrame,
        ground_truth_positive_pair: ps.DataFrame,
        ground_truth_negative_pair: ps.DataFrame,
    ) -> SamplingResult:
        """
        Sample the knowledge graph, ground truth, and embeddings.

        Args:
            knowledge_graph_nodes: Knowledge graph nodes
            knowledge_graph_edges: Knowledge graph edges
            ground_truth_positive_pair: Ground truth positive pairs
            ground_truth_negative_pair: Ground truth negative pairs

        Returns:
            SamplingResult with fields:
                knowledge_graph_nodes: Sampled knowledge graph nodes
                knowledge_graph_edges: Sampled knowledge graph edges
        """


class GroundTruthSampler(Sampler):
    """
    A sampler that will sample from ground truth positives and negatives pairs, and then from the rest of the graph.

    The aim is to keep the proportion of ground truth nodes in the output.
    """

    def __init__(
        self,
        ground_truth_positive_sample_ratio: float,
        ground_truth_negative_sample_ratio: float,
        knowledge_graph_nodes_sample_ratio: float,
        seed: int,
    ):
        self.ground_truth_positive_sample_ratio = ground_truth_positive_sample_ratio
        self.ground_truth_negative_sample_ratio = ground_truth_negative_sample_ratio
        self.knowledge_graph_nodes_sample_ratio = knowledge_graph_nodes_sample_ratio
        self.seed = seed

    def sample(
        self,
        knowledge_graph_nodes: ps.DataFrame,
        knowledge_graph_edges: ps.DataFrame,
        ground_truth_positive_pair: ps.DataFrame,
        ground_truth_negative_pair: ps.DataFrame,
    ) -> SamplingResult:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Received {knowledge_graph_nodes.count()} knowledge graph nodes")
            logger.info(f"Received {knowledge_graph_edges.count()} knowledge graph edges")
            logger.info(f"Received {ground_truth_positive_pair.count()} ground truth positives")
            logger.info(f"Received {ground_truth_negative_pair.count()} ground truth negatives")

        # ... select ground truth pairs that are in the knowledge graph
        ground_truth_positive_pair_in_knowledge_graph = (
            ground_truth_positive_pair.join(
                knowledge_graph_nodes.alias("source_nodes"),
                ground_truth_positive_pair.source == f.col("source_nodes.id"),
            )
            .join(
                knowledge_graph_nodes.alias("target_nodes"),
                ground_truth_positive_pair.target == f.col("target_nodes.id"),
            )
            .select(ground_truth_positive_pair["*"])
        )
        ground_truth_negative_pair_in_knowledge_graph = (
            ground_truth_negative_pair.join(
                knowledge_graph_nodes.alias("source_nodes"),
                ground_truth_negative_pair.source == f.col("source_nodes.id"),
            )
            .join(
                knowledge_graph_nodes.alias("target_nodes"),
                ground_truth_negative_pair.target == f.col("target_nodes.id"),
            )
            .select(ground_truth_negative_pair["*"])
        )

        # ... sample from ground truth pairs
        sampled_ground_truth_positive_pair = ground_truth_positive_pair_in_knowledge_graph.sample(
            self.ground_truth_positive_sample_ratio, self.seed
        )
        sampled_ground_truth_negative_pair = ground_truth_negative_pair_in_knowledge_graph.sample(
            self.ground_truth_negative_sample_ratio, self.seed
        )

        # ... select nodes attached to sampled ground truth pairs
        all_sampled_ground_truth = sampled_ground_truth_positive_pair.union(sampled_ground_truth_negative_pair)
        sampled_ground_truth_node_ids = (
            all_sampled_ground_truth.withColumn("ids", f.array("source", "target"))
            .select(f.explode("ids").alias("id"))
            .distinct()
        )

        # ... sample from all KG nodes
        sampled_knowledge_graph_node_ids = knowledge_graph_nodes.select("id").sample(
            self.knowledge_graph_nodes_sample_ratio, self.seed
        )

        # ... aggregate nodes and drop duplicated node ids
        sampled_node_ids = sampled_knowledge_graph_node_ids.union(sampled_ground_truth_node_ids).distinct()

        # ... join on knowledge graph nodes
        sampled_nodes = (
            knowledge_graph_nodes.alias("knowledge_graph_nodes")
            .join(sampled_node_ids, sampled_node_ids.id == knowledge_graph_nodes.id, "inner")
            .select("knowledge_graph_nodes.*")
        )

        # ... join on knowledge graph edges
        sampled_edges = (
            knowledge_graph_edges.alias("knowledge_graph_edges")
            .join(sampled_node_ids, sampled_node_ids.id == knowledge_graph_edges.source, "inner")
            .select("knowledge_graph_edges.*")
            .join(sampled_node_ids, sampled_node_ids.id == knowledge_graph_edges.target, "inner")
            .select("knowledge_graph_edges.*")
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Sampled {sampled_ground_truth_positive_pair.count()} ground truth positives")
            logger.info(f"Sampled {sampled_ground_truth_negative_pair.count()} ground truth negatives")
            logger.info(f"Sampled {sampled_ground_truth_node_ids.count()} sampled unique nodes from ground truth")
            logger.info(f"Sampled {sampled_knowledge_graph_node_ids.count()} knowledge graph nodes")
            logger.info(f"Sampled {sampled_nodes.cache().count()} sampled nodes in total")
            logger.info(f"Sampled {sampled_edges.cache().count()} sampled edges in total")

        return SamplingResult(sampled_nodes, sampled_edges)
