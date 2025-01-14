import logging

import pyspark.sql as ps

logger = logging.getLogger(__name__)


def sample_kg(nodes: ps.DataFrame, edges: ps.DataFrame, top_nodes: int, **kwargs):
    logger.info(f"Received {nodes.count()} nodes and {edges.count()} edges as an input")

    sampled_nodes = nodes.limit(top_nodes)
    sampled_edges_subject = edges.join(sampled_nodes.select("id"), edges.subject == sampled_nodes.id, "inner").select(
        *edges.columns
    )
    sampled_edges_object = edges.join(sampled_nodes.select("id"), edges.object == sampled_nodes.id, "inner").select(
        *edges.columns
    )
    sampled_edges = sampled_edges_subject.union(sampled_edges_object).distinct()

    logger.info(f"Sampled {sampled_nodes.count()} nodes and {sampled_edges.count()} edges")
    return (sampled_nodes, sampled_edges)
