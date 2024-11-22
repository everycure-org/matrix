import pyspark as ps
from pyspark.sql.functions import col, split

"""
Sprint Goal
Implement tracking for following metrics:
 # of nodes ingested
 # of nodes normalized (via Node Normalizer)
 # of nodes that failed normalization
 Node to Normalized Node map
 # of edges ingested
 # of edges that are valid (Biolink predicate with subject, object, knowledge level, agent type, and primary knowledge source)
 # of edges not valid
 Log of invalid edges and why they were flagged as invalid
 Count of node types and edge types
 Track node-edge-node relationships (count Subject-Predicate-Object triples)
 Identify unconnected nodes and nodes removed post-normalization

EKC notes:
It looks like Pascal implemented the following nodes.py in this directory:
 Node to Normalized Node map
 He made a mapping_df but I don't know where that is. 

"""

# Writing something basic, but can't yet try it.
# need to unlock git-crypt
# need to understand how to run up to integration in kedro.


def metrics(
    nodes: ps.sql.DataFrame, edges: ps.sql.DataFrame, norm_nodes: ps.sql.DataFrame, norm_edges: ps.sql.DataFrame
) -> ps.sql.DataFrame:
    nodes_count = nodes.count()
    print(f"Number of nodes before normalization: {nodes_count}")

    norm_nodes_count = norm_nodes.count()
    print(f"Number of nodes after normalization: {norm_nodes_count}")

    print(f"Number of nodes that failed normalization: {nodes_count - norm_nodes_count}")

    edges_count = edges.count()
    print(f"Number of nodes before normalization: {edges_count}")

    norm_edges_count = norm_edges.count()
    print(f"Number of nodes after normalization: {norm_edges_count}")

    print(f"Number of nodes that failed normalization: {edges_count - norm_edges_count}")

    nodes_with_prefix = nodes.withColumn("prefix", split(col("id"), ":")[0])
    prefix_counts = nodes_with_prefix.groupBy("prefix").count()

    print(f"Types of nodes: {prefix_counts}")
