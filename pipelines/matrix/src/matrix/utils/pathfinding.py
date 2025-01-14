"""Module for path-finding between pairs of nodes in a knowledge graph.

Based off the "meet in the middle" approach, whereby we find outgoing paths for the source and target nodes, then join them in the middle.
Currently only supports paths up to 4 hops, but this can be easily extended.
"""

from functools import partial
from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def preprocess_edges(
    edges: DataFrame,
    nodes: DataFrame,
    node_attribute_columns: List[str] = ["name", "category"],
) -> DataFrame:
    """Get a dataframe describing a decorated undirected graph from directed knowledge graph.

    Args:
        edges: Dataframe containing the edges of a directed knowledge graph.
            Must have columns subject, object and predicate.
        nodes: Dataframe containing the nodes of the knowledge graph.
            Must have column id corresponding to subject and object in edges.
        node_attribute_columns: List of columns in the nodes dataframe to be added to the preprocessed edges dataframe.

    Returns:
        Dataframe containing the corresponding undirected graph with:
            - Aggregated "predicates" flags for the edges in the form of a list of strings.
                NOTE: Predicates for backwards edges are prefixed with "*".
            - Directionality flags "is_forward" and "is_backward" for the edges.
            - Node attributes for the target nodes.
    """
    # Decorate edges with attributes and is_forward flag
    edges_decorated = edges.select("subject", "object", "predicate").withColumn("is_forward", F.lit(True))

    # Collect multiple predicates into lists
    edges_decorated = edges_decorated.groupBy("subject", "object").agg(
        F.max("is_forward").alias("is_forward"), F.collect_list("predicate").alias("predicates")
    )

    # Get flipped edges and decorate with attributes and is_backward flag
    edges_flipped_decorated = (
        edges.withColumnsRenamed({"subject": "o", "object": "s"})
        .withColumnsRenamed({"o": "object", "s": "subject"})
        .select("subject", "object", "predicate")
        .withColumn("is_backward", F.lit(True))
    )

    # Collect multiple predicates into lists
    edges_flipped_decorated = (
        edges_flipped_decorated.withColumn("predicate", F.concat(F.lit("*"), F.col("predicate")))
        .groupBy("subject", "object")
        .agg(
            F.collect_list("predicate").alias("predicates"),
            F.max("is_backward").alias("is_backward"),
        )
    )

    # Union aggregated edges and flipped edges
    edges_union = (
        edges_decorated.select("subject", "object", "predicates")
        .unionByName(edges_flipped_decorated.select("subject", "object", "predicates"))  # Union edges and flipped edges
        .groupBy("subject", "object")
        .agg(F.flatten(F.collect_list("predicates")).alias("predicates"))  # Collect forward and backward predicates
        .join(
            edges_decorated.select("subject", "object", "is_forward"), on=["subject", "object"], how="outer"
        )  # Add back in is_forward flag
        .fillna(False, subset=["is_forward"])
        .join(
            edges_flipped_decorated.select("subject", "object", "is_backward"), on=["subject", "object"], how="outer"
        )  # Add back in is_backward flag
        .fillna(False, subset=["is_backward"])
    )

    # Add in node attributes and return
    return edges_union.join(
        nodes.select("id", *node_attribute_columns)
        .withColumnRenamed("id", "object")
        .withColumnsRenamed({node_attribute: "object_" + node_attribute for node_attribute in node_attribute_columns}),
        on=["object"],
        how="left",
    )


def get_neighbours_join(
    start_nodes: DataFrame,
    preprocessed_edges: DataFrame,
    source_column: str = "start_node",
    neighbour_column: str = "neighbour",
    decorate_neighbours: bool = True,
    edge_attribute_prefix: str = "hop_",
    node_attribute_columns: List[str] = ["name", "category"],
) -> DataFrame:
    """Get the neighbours of a given set of nodes.

    The corresponding edges are decorated with directionality and predicates.
    Optionally, the neighbours are decorated with specified attributes.

    FUTURE: The dataframe start_nodes may contain multiple rows for the same node.
        (i.e. the same value in source_column but varying values in other columns).
        We expect this to happen often in path-finding tasks.
        Can we take advantage of this to get a speed boost, using a well placed .distinct()?

    Args:
        start_nodes: Dataframe containing the source nodes.
        preprocessed_edges: Dataframe containing a decorated undirected graph.
        source_column: Name of the column in start_nodes containing the source nodes.
        neighbour_column: Name of the column in the output DataFrame containing the neighbours.
        decorate_neighbours: Determines if neighbour nodes are decorated with attributes.
        edge_attribute_prefix: prefix for edge attribute columns
        node_attribute_columns: list of node attribute columns

    Returns:
        Dataframe containing all start nodes with their neighbours. Each row contains a start node and a single neighbour.
    """
    # Collect node and edge attribute columns coming along for the ride
    along_for_the_ride = ["predicates", "is_forward", "is_backward"]
    edge_attribute_renames = {
        edge_attribute: edge_attribute_prefix + edge_attribute for edge_attribute in along_for_the_ride
    }
    node_attribute_renames = {
        "object_" + node_attribute: neighbour_column + "_" + node_attribute for node_attribute in node_attribute_columns
    }

    # Prepare column renames for the join operation
    column_renames_for_join = {"subject": source_column, "object": neighbour_column} | edge_attribute_renames
    if decorate_neighbours:
        column_renames_for_join = column_renames_for_join | node_attribute_renames

    # Perform join and return
    return start_nodes.join(
        preprocessed_edges.withColumnsRenamed(column_renames_for_join), on=source_column, how="inner"
    )


def _process_paths(
    paths: DataFrame,
    n_hops: int,
    node_index_lst: List[str],
    hop_index_lst: List[str],
    node_attributes: List[str],
) -> DataFrame:
    """
    Transforms paths datasets into desired schema.

    For use in the "meet in the middle" path-finding algorithm.

    e.g. Node IDs in a two-hop paths:
    - Before transformation: represented by several columns (e.g. node_0, node_1, node_m0)
    - After transformation: represented by a single column (e.g. node_lst)

    Args:
        paths: Input dataframe containing the paths.
        n_hops: Number of hops in the paths.
        node_index_lst: List of node indices in the paths (e.g. ["0", "1", "m0"] for two-hop paths).
        hop_index_lst: List of hop indices in the paths (e.g. ["1", "m1"] for two-hop paths).
        node_attributes: List of node attributes to be collected.

    """
    # Fixed edges attributes
    edge_attributes = ["predicates", "is_forward", "is_backward"]

    # Schema for output dataframe
    final_schema = (
        ["source", "target", "n_hops", "node_lst"]
        + ["hop_" + attribute + "_lst" for attribute in edge_attributes]
        + ["node_" + attribute + "_lst" for attribute in node_attributes]
    )

    paths_processed = (
        paths.withColumns(
            {
                "n_hops": F.lit(n_hops),
                "node_lst": F.array(*["node_" + index for index in node_index_lst]),  # Collect all node IDs
                **{  # Collect edge attributes for all hops
                    "hop_" + attribute + "_lst": F.array(*["hop_" + index + f"_{attribute}" for index in hop_index_lst])
                    for attribute in edge_attributes
                },
                **{  # Collect other node attributes
                    "node_" + attribute + "_lst": F.array(
                        *["node_" + index + f"_{attribute}" for index in node_index_lst]
                    )
                    for attribute in node_attributes
                },
            }
        )
        .withColumnsRenamed(
            {  # Collect source and target nodes
                "node_" + node_index_lst[0]: "source",
                "node_" + node_index_lst[-1]: "target",
            }
        )
        .select(*final_schema)
    )

    return paths_processed


def get_connecting_paths(
    input_pairs: DataFrame,
    preprocessed_edges: DataFrame,
    nodes: DataFrame,
    n_hops: int,
    source_column: str = "source",
    target_column: str = "target",
    node_attribute_columns: List[str] = ["name", "category"],
) -> DataFrame:
    """
    Get connecting paths for a given set of input pairs up to n_hops <= 4.

    The algorithm is based off the "meet in the middle" approach.
    Outgoing paths are computed for the source and target nodes, which are then joined to find the connecting paths.
    The logic is illustrated by the following diagram:
        2-hop: (source) node_0 -- node_1 = node_m1 -- node_m0 (target)
        3-hop: (source) node_0 -- node_1 -- node_2 = node_m1 -- node_m0 (target)
        4-hop: (source) node_0 -- node_1 -- node_2 = node_m2 -- node_m1 -- node_m0 (target)

    FUTURE: The algorithm is currently hardcoded for n_hops <= 4 but can be easily extended to higher n_hops.

    Args:
        input_pairs: Input dataframe containing the source and target nodes.
        preprocessed_edges: Dataframe containing a decorated undirected graph.
        nodes: Dataframe containing the nodes of the knowledge graph.
        n_hops: Maximum number of hops to compute. Must be equal to 2, 3 or 4.
        source_column: Name of the column in input_pairs containing the source nodes.
        target_column: Name of the column in input_pairs containing the target nodes.
        node_attribute_columns: List of node attribute columns.

    Returns:
        Dataframe containing all connecting paths for a minimum hop length of 2 and a maximum hop length of n_hops.
    """
    f_neighbours = partial(
        get_neighbours_join, preprocessed_edges=preprocessed_edges, node_attribute_columns=node_attribute_columns
    )

    # Add pair ID to input pairs (used for the "middle" join operation)
    pairs_with_id = input_pairs.withColumn("pair_ID", F.monotonically_increasing_id())

    # Get source and target nodes and enrich with node attributes
    source_nodes = (
        pairs_with_id.select(source_column, "pair_ID")  # Restrict
        .join(  # Enrich
            nodes.select("id", *node_attribute_columns).withColumnRenamed("id", source_column),
            on=source_column,
            how="left",
        )
        .withColumnsRenamed(
            {  # Organise schema
                source_column: "node_0",
                **{node_attribute: "node_0_" + node_attribute for node_attribute in node_attribute_columns},
            }
        )
    )

    target_nodes = (  # Restrict
        pairs_with_id.select(target_column, "pair_ID")
        .join(  # Enrich
            nodes.select("id", *node_attribute_columns).withColumnRenamed("id", target_column),
            on=target_column,
            how="left",
        )
        .withColumnsRenamed(
            {  # Organise schema
                target_column: "node_m0",
                **{node_attribute: "node_m0_" + node_attribute for node_attribute in node_attribute_columns},
            }
        )
    )

    # Get outgoing paths for source

    source_paths_1 = f_neighbours(
        source_nodes, source_column="node_0", neighbour_column="node_1", edge_attribute_prefix="hop_1_"
    )

    if n_hops == 3 or n_hops == 4:
        source_paths_2 = f_neighbours(
            source_paths_1, source_column="node_1", neighbour_column="node_2", edge_attribute_prefix="hop_2_"
        )

    # Get outgoing paths for target (NOTE: decorate_neighbours argument suppresses node decoration on the join node to avoid data duplication)

    if n_hops == 2 or n_hops == 3:
        target_paths_1 = f_neighbours(
            target_nodes,
            source_column="node_m0",
            neighbour_column="node_m1",
            edge_attribute_prefix="hop_m1_",
            decorate_neighbours=False,
        )

    if n_hops == 4:
        target_paths_1 = f_neighbours(
            target_nodes, source_column="node_m0", neighbour_column="node_m1", edge_attribute_prefix="hop_m1_"
        )
        target_paths_2 = f_neighbours(
            target_paths_1,
            source_column="node_m1",
            neighbour_column="node_m2",
            edge_attribute_prefix="hop_m2_",
            decorate_neighbours=False,
        )

    # Perform big join in the middle

    two_hop_paths = source_paths_1.join(
        target_paths_1.withColumnRenamed("node_m1", "node_1"), on=["node_1", "pair_ID"], how="inner"
    )

    if n_hops == 3 or n_hops == 4:
        three_hop_paths = source_paths_2.join(
            target_paths_1.withColumnRenamed("node_m1", "node_2"), on=["node_2", "pair_ID"], how="inner"
        )

    if n_hops == 4:
        four_hop_paths = source_paths_2.join(
            target_paths_2.withColumnRenamed("node_m2", "node_2"), on=["node_2", "pair_ID"], how="inner"
        )

    # Process paths and combine into a single dataframe
    process_paths = partial(_process_paths, node_attributes=node_attribute_columns)

    two_hop_paths_processed = process_paths(
        paths=two_hop_paths, n_hops=2, node_index_lst=["0", "1", "m0"], hop_index_lst=["1", "m1"]
    )

    if n_hops == 2:
        return two_hop_paths_processed

    three_hop_paths_processed = process_paths(
        paths=three_hop_paths, n_hops=3, node_index_lst=["0", "1", "2", "m0"], hop_index_lst=["1", "2", "m1"]
    )
    paths_combined = two_hop_paths_processed.union(three_hop_paths_processed)

    if n_hops == 3:
        return paths_combined

    four_hop_paths_processed = process_paths(
        paths=four_hop_paths, n_hops=4, node_index_lst=["0", "1", "2", "m0", "m1"], hop_index_lst=["1", "2", "m1", "m2"]
    )
    paths_combined = paths_combined.union(four_hop_paths_processed)

    return paths_combined
