import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import pandas as pd
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
    ) -> SamplingResult:
        """
        Args:
            knowledge_graph_nodes: Knowledge graph nodes
            knowledge_graph_edges: Knowledge graph edges

        Returns:
            SamplingResult with fields:
                knowledge_graph_nodes: Sampled knowledge graph nodes
                knowledge_graph_edges: Sampled knowledge graph edges
        """


class KnowledgeGraphSampler(Sampler):
    """
    A sampler that will sample from knowledge graph nodes, and then only keep edges between those sampled nodes.
    """

    def __init__(
        self,
        knowledge_graph_nodes_sample_ratio: float,
        seed: int,
    ):
        self.knowledge_graph_nodes_sample_ratio = knowledge_graph_nodes_sample_ratio
        self.seed = seed

    def sample(
        self,
        knowledge_graph_nodes: ps.DataFrame,
        knowledge_graph_edges: ps.DataFrame,
    ) -> SamplingResult:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Received {knowledge_graph_nodes.count()} knowledge graph nodes")
            logger.info(f"Received {knowledge_graph_edges.count()} knowledge graph edges")

        # ... sample from all knowledge graph nodes
        sampled_nodes = knowledge_graph_nodes.sample(self.knowledge_graph_nodes_sample_ratio, self.seed)

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

        return SamplingResult(sampled_nodes, sampled_edges)
