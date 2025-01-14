import logging

import pyspark.sql as ps

logger = logging.getLogger(__name__)


def sample_kg(nodes: ps.DataFrame, edges: ps.DataFrame, **kwargs):
    logger.info(f"Received {nodes.count()} nodes and {edges.count()} edges as an input")

    sampled_nodes_count = 500
    sampled_nodes = nodes.limit(sampled_nodes_count)
    sampled_edges_subject = edges.join(sampled_nodes.select("id"), edges.subject == sampled_nodes.id, "inner").select(
        *edges.columns
    )
    sampled_edges_object = edges.join(sampled_nodes.select("id"), edges.object == sampled_nodes.id, "inner").select(
        *edges.columns
    )
    sampled_edges = sampled_edges_subject.union(sampled_edges_object).distinct()

    logger.info(f"Sampled nodes: {sampled_nodes.head(5)}")
    logger.info(f"Sampled edges: {sampled_edges.head(5)}")
    return (sampled_nodes, sampled_edges)
