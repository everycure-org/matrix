import logging
from abc import ABC, abstractmethod
from typing import Dict

import pyspark.sql as ps
import pyspark.sql.functions as f

logger = logging.getLogger(__name__)


class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        knowledge_graph_nodes: ps.DataFrame,
        knowledge_graph_edges: ps.DataFrame,
        ground_truth_edges: ps.DataFrame,
    ) -> Dict[str, ps.DataFrame]:
        """
        Args:
            knowledge_graph_nodes: Knowledge graph nodes
            knowledge_graph_edges: Knowledge graph edges
            ground_truth_edges: Ground truth edges

        Returns:
            SamplingResult with fields:
                knowledge_graph_nodes: Sampled knowledge graph nodes
                knowledge_graph_edges: Sampled knowledge graph edges
        """


class GroundTruthRandomSampler(Sampler):
    """
    A sampler that will sample from knowledge graph nodes, and then only keep edges between those sampled nodes.
    """

    def __init__(
        self,
        knowledge_graph_nodes_sample_ratio: float,
        ground_truth_edges_sample_ratio: float,
        seed: int,
    ):
        self.knowledge_graph_nodes_sample_ratio = knowledge_graph_nodes_sample_ratio
        self.ground_truth_edges_sample_ratio = ground_truth_edges_sample_ratio
        self.seed = seed

    def sample(
        self,
        knowledge_graph_nodes: ps.DataFrame,
        knowledge_graph_edges: ps.DataFrame,
        ground_truth_edges: ps.DataFrame,
    ) -> Dict[str, ps.DataFrame]:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Received {knowledge_graph_nodes.count()} knowledge graph nodes")
            logger.info(f"Received {knowledge_graph_edges.count()} knowledge graph edges")
            logger.info(f"Received {ground_truth_edges.count()} ground truth edges")

        # ... sample ground truth pairs and extract node ids
        ground_truth_node_ids = (
            ground_truth_edges.sampleBy(
                "y", {0: self.ground_truth_edges_sample_ratio, 1: self.ground_truth_edges_sample_ratio}, self.seed
            )
            .withColumn("ids", f.array("object", "subject"))
            .select(f.explode("ids").alias("id"))
            .distinct()
        )

        # ... join ground  truth nodes back to knowledge graph nodes
        ground_truth_sampled_nodes = ground_truth_node_ids.join(
            knowledge_graph_nodes.alias("knowledge_graph_nodes"),
            ground_truth_node_ids.id == knowledge_graph_nodes.id,
            "inner",
        ).select("knowledge_graph_nodes.*")

        # ... sample from all knowledge graph nodes
        knowledge_graph_sampled_nodes = knowledge_graph_nodes.sample(self.knowledge_graph_nodes_sample_ratio, self.seed)

        # ... union all sampled nodes
        sampled_nodes = ground_truth_sampled_nodes.union(knowledge_graph_sampled_nodes).distinct()

        # ... join on knowledge graph edges
        sampled_edges = (
            knowledge_graph_edges.alias("knowledge_graph_edges")
            .join(sampled_nodes, sampled_nodes.id == knowledge_graph_edges.object, "inner")
            .select("knowledge_graph_edges.*")
            .join(sampled_nodes, sampled_nodes.id == knowledge_graph_edges.subject, "inner")
            .select("knowledge_graph_edges.*")
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Sampled {sampled_nodes.cache().count()} sampled nodes in total")
            logger.info(f"Sampled {sampled_edges.cache().count()} sampled edges in total")

        return {
            "nodes": sampled_nodes,
            "edges": sampled_edges,
        }
