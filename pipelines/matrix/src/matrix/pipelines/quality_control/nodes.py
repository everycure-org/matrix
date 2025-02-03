from typing import Dict, List

import pyspark.sql as ps
import pyspark.sql.functions as F


def count_nodes_by(nodes: ps.DataFrame) -> ps.DataFrame:
    count_by_columns: List[str] = ["category", "prefix", "upstream_data_source"]
    return (
        nodes.withColumn("prefix", F.split("id", ":")[0]).select(*count_by_columns).groupBy(*count_by_columns).count()
    )


def count_edges_by(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    subject_nodes = (
        nodes.select("id", "category").withColumnRenamed("category", "subject_category").alias("subject_nodes")
    )
    object_nodes = nodes.select("id", "category").withColumnRenamed("category", "object_category").alias("object_nodes")

    count_by_columns = [
        "subject_prefix",
        "subject_category",
        "predicate",
        "object_prefix",
        "object_category",
        "primary_knowledge_source",
        "aggregator_knowledge_source",
        "upstream_data_source",
    ]

    return (
        edges.withColumn("subject_prefix", F.split("subject", ":")[0])
        .withColumn("object_prefix", F.split("object", ":")[0])
        .join(subject_nodes, edges.subject == subject_nodes.id, "left")
        .join(object_nodes, edges.object == object_nodes.id, "left")
        .select(*count_by_columns)
        .groupBy(*count_by_columns)
        .count()
    )
