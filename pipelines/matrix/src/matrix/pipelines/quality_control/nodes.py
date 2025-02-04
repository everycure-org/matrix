from typing import Dict

import pyspark.sql as ps
import pyspark.sql.functions as F

from matrix.inject import inject_object
from matrix.pipelines.integration.transformers.transformer import GraphTransformer


@inject_object()
def count_untransformed_knowledge_graph(
    transformer: GraphTransformer, nodes: ps.DataFrame, edges: ps.DataFrame
) -> Dict[str, ps.DataFrame]:
    return transformer.count_knowledge_graph(nodes, edges)


def _count_nodes(nodes: ps.DataFrame) -> ps.DataFrame:
    nodes_count_by_columns = ["category", "prefix", "upstream_data_source"]
    nodes_report = (
        nodes.withColumn("prefix", F.split("id", ":")[0])
        .select(*nodes_count_by_columns)
        .groupBy(*nodes_count_by_columns)
        .count()
    )
    return nodes_report


def _count_edges(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    subject_nodes = (
        nodes.select("id", "category").withColumnRenamed("category", "subject_category").alias("subject_nodes")
    )
    object_nodes = nodes.select("id", "category").withColumnRenamed("category", "object_category").alias("object_nodes")

    edges_count_by_columns = [
        "subject_prefix",
        "subject_category",
        "predicate",
        "object_prefix",
        "object_category",
        "primary_knowledge_source",
        "aggregator_knowledge_source",
        "upstream_data_source",
    ]
    edges_report = (
        edges.withColumn("subject_prefix", F.split("subject", ":")[0])
        .withColumn("object_prefix", F.split("object", ":")[0])
        .join(subject_nodes, edges.subject == subject_nodes.id, "left")
        .join(object_nodes, edges.object == object_nodes.id, "left")
        .select(*edges_count_by_columns)
        .groupBy(*edges_count_by_columns)
        .count()
    )

    return edges_report


def count_knowledge_graph(nodes: ps.DataFrame, edges: ps.DataFrame) -> Dict[str, ps.DataFrame]:
    nodes_report = _count_nodes(nodes)
    edges_report = _count_edges(nodes, edges)
    return {
        "nodes_report": nodes_report,
        "edges_report": edges_report,
    }
