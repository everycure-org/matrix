import pyspark as ps
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

"""
Sprint Goal
Implement tracking for following metrics:
[x] # of nodes ingested
[ ] # of nodes normalized (via Node Normalizer)
[ ] # of nodes that failed normalization
[ ] Node to Normalized Node map
[x] # of edges ingested
[ ] # of edges that are valid (Biolink predicate with subject, object, knowledge level, agent type, and primary knowledge source)
[ ] # of edges not valid
[ ] Log of invalid edges and why they were flagged as invalid
[x] Count of node types and edge types
[ ] Track node-edge-node relationships (count Subject-Predicate-Object triples)
[ ] Identify unconnected nodes and nodes removed post-normalization

EKC notes:
It looks like Pascal implemented the following nodes.py in this directory:
 Node to Normalized Node map
 He made a mapping_df but I don't know where that is. 

"""

# Writing something basic, but can't yet try it.
# need to unlock git-crypt
# need to understand how to run up to integration in kedro.


def ingestion(
    nodes: ps.sql.DataFrame,
    edges: ps.sql.DataFrame,
    dataset_name: str,
) -> DataFrame:
    source = dataset_name
    nodes_count = nodes.count()

    node_prefix = nodes.withColumn("prefix", F.split(F.col("id:ID"), ":")[0])
    prefix_counts = node_prefix.groupBy("prefix").count()
    prefix_counts_dict = {row["prefix"]: row["count"] for row in prefix_counts.collect()}

    edges_count = edges.count()

    summary_data = pd.DataFrame(
        [
            {"metric": f"total number of nodes in {source}", "value": nodes_count},
            {"metric": f"number of unique node types {source}", "value": len(prefix_counts_dict)},
            {"metric": f"number of each type of node {source}", "value": str(prefix_counts_dict)},
            {"metric": f"number of edges {source}", "value": edges_count},
        ]
    )
    spark = ps.sql.SparkSession.builder.getOrCreate()
    summary_spark_df = spark.createDataFrame(summary_data)

    return summary_spark_df
