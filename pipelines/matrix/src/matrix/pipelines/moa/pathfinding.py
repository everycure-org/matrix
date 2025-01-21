"""Module containing utilities for path-finding in knowledge-graphs"""

import math
from functools import partial
from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def _flip_edges(edges: DataFrame) -> DataFrame:
    """Flip the edges of a dataframe.

    Args:
        edges: Dataframe containing the edges of the directed knowledge graph.

    Returns:
        Dataframe with the flipped edges.
    """
    return edges.withColumnsRenamed({"subject": "o", "object": "s"}).withColumnsRenamed({"o": "object", "s": "subject"})


def _collect_attributes_as_dict(edges: DataFrame, attribute_column_list: List[str]) -> DataFrame:
    """Collect attributes as a dictionary.

    Args:
        edges: Dataframe containing the edges of the directed knowledge graph.
        attribute_column_list: List of columns in edges containing the attributes.

    Returns:
        Dataframe with the attribute collected as a dictionary.
    """
    return edges.withColumn(
        "edge_attribute",
        F.create_map(
            *[x for attribute in attribute_column_list for x in [F.lit(attribute), F.col(attribute)]],
        ),
    ).select("subject", "object", "edge_attribute")


def preprocess_edges(
    edges: DataFrame,
    attribute_column_list: List[str] = ["predicate"],
) -> DataFrame:
    """Preprocesses the edges of a directed graph for path-finding.

    Adds in the flipped edges and decorates with attributes to keep track of the directionality of the original edges.
    This allows us to traverse each edge in both directions during path-finding.

    Args:
        edges: Dataframe containing the edges of the directed knowledge graph.
            Must have columns "subject", "object".
        attribute_column_list: List of column names in edges to be collected as attributes.

    Returns:
        Dataframe with columns "subject", "object", "attributes", where "attributes"
        is a list of dictionaries containing all possible attributes (including directionality) for the subject-object-pair.
    """
    return (
        edges.withColumn("is_forward", F.lit(True).cast("string"))
        .unionByName(edges.transform(_flip_edges).withColumn("is_forward", F.lit(False).cast("string")))
        .transform(_collect_attributes_as_dict, [*attribute_column_list, "is_forward"])
        .groupBy("subject", "object")
        .agg(F.collect_list("edge_attribute").alias("edge_attributes"))
    )


def get_neighbours_join(
    start_nodes: DataFrame,
    preprocessed_edges: DataFrame,
    start_node_column: str = "start_node",
    neighbour_column: str = "neighbour",
) -> DataFrame:
    """Get the neighbours of a given set of nodes.

    Example:Suppose that node_1 and node_2 have two neighbours each. Then:
        Input DataFrame:
        | start_node |
        |------------|
        | node_1    |
        | node_2    |

        Output DataFrame:
        | start_node | neighbour     |
        |------------|---------------|
        | node_1    | neighbour_1_1 |
        | node_1    | neighbour_1_2 |
        | node_2    | neighbour_2_1 |
        | node_2    | neighbour_2_2 |

    Args:
        start_nodes: Dataframe containing the source nodes.
        preprocessed_edges: Preprocessed edges dataframe.
        start_node_column: Name of the column in start_nodes containing the source nodes.
        neighbour_column: Name of the column in the output DataFrame containing the neighbours.

    Returns:
        Dataframe containing all start nodes with their neighbours. Each row contains a start node and a single neighbour.
    """
    # Perform join and return
    return start_nodes.join(
        preprocessed_edges.select("subject", "object").withColumnsRenamed(
            {"subject": start_node_column, "object": neighbour_column}
        ),
        on=start_node_column,
        how="inner",
    )


def _process_paths(paths: DataFrame, n_hops: int) -> DataFrame:
    """Process paths dataframe.

    Args:
        paths: Dataframe with columns "node_0", ..., f"node_{n_hops}" describing a set of paths.
        n_hops: The length of the paths.

    Returns:
        Dataframe with columns "source", "target", "nodes_list", where "nodes_list" is a list of nodes in the path.
    """
    return paths.withColumns(
        {
            "source": F.col("node_0"),
            "target": F.col(f"node_{n_hops}"),
            "node_id_list": F.array([F.col(f"node_{i}") for i in range(n_hops + 1)]),
        }
    ).select("source", "target", "node_id_list")


def get_connecting_paths(
    input_pairs: DataFrame,
    preprocessed_edges: DataFrame,
    n_hops: int,
    source_column: str = "source",
    target_column: str = "target",
    no_repeats: bool = True,
) -> DataFrame:
    """Get all connecting paths of hop length n_hops for a set of input pairs.

    Args:
        input_pairs: Dataframe containing the source and target nodes describing a set of input pairs.
        preprocessed_edges: Preprocessed edges dataframe.
        n_hops: The length of the paths to be returned.
        source_column: The name of the column in input_pairs containing the source nodes.
        target_column: The name of the column in input_pairs containing the target nodes.
        no_repeats: Boolean flag indicating whether to filter out paths with repeated nodes.

    Returns:
        Dataframe with columns "source", "target", "nodes_list" describing all paths for all input pairs.
        Here, "nodes_list" is a list of nodes in the path.
    """
    # Add id for each pair
    input_pairs = input_pairs.withColumn("pair_id", F.monotonically_increasing_id())

    # Split hops between source and target
    n_hops_source = math.ceil(n_hops / 2)
    n_hops_target = n_hops - n_hops_source

    # Get outgoing paths for source nodes
    source_paths = (
        input_pairs.select(source_column)
        .distinct()  # Avoid repeat computations
        .withColumnRenamed(source_column, "node_0")
    )
    for i in range(n_hops_source):
        source_paths = get_neighbours_join(
            source_paths,
            preprocessed_edges,
            start_node_column=f"node_{i}",
            neighbour_column=f"node_{i + 1}",
        )
    source_paths = source_paths.join(  # Add back in pair_id
        input_pairs.select("pair_id", source_column).withColumnRenamed(source_column, "node_0"),
        on=["node_0"],
        how="inner",
    )

    # Get outgoing paths for target nodes
    target_paths = (
        input_pairs.select(target_column)
        .distinct()  # Avoid repeat computations
        .withColumnRenamed(target_column, f"node_{n_hops}")
    )
    for i in range(n_hops_target):
        target_paths = get_neighbours_join(
            target_paths,
            preprocessed_edges,
            start_node_column=f"node_{n_hops - i}",
            neighbour_column=f"node_{n_hops - i - 1}",
        )
    target_paths = target_paths.join(  # Add back in pair_id
        input_pairs.select("pair_id", target_column).withColumnRenamed(target_column, f"node_{n_hops}"),
        on=[f"node_{n_hops}"],
        how="inner",
    )

    # Join outgoing source and target paths in the middle for each pair
    paths = source_paths.join(target_paths, on=[f"node_{n_hops_source}", "pair_id"], how="inner")

    # Collect nodes columns into a single array
    processed_paths = _process_paths(paths, n_hops)

    # Remove paths with repeated nodes
    if no_repeats:
        processed_paths = processed_paths.filter(F.size(F.array_distinct(F.col("node_id_list"))) == n_hops + 1)

    return processed_paths


def enrich_paths_with_node_attributes(
    paths: DataFrame,
    nodes: DataFrame,
    node_attribute_list: List[str],
) -> DataFrame:
    """Enrich a paths dataframe with node attribute information.

    Args:
        paths: Dataframe describing a set of paths containing a "nodes_list" column.
        nodes: Dataframe describing a set of nodes and their attributes containing a "id" column.
        node_attribute_list: List of node attributes columns from nodes.

    Returns:
        Dataframe with the same columns as paths, but with additional array-valued columns for each node attribute.
    """
    # Add unique ID for each path
    paths_with_id = paths.withColumn("path_id", F.monotonically_increasing_id())

    # Explode paths into hops so that each row corresponds to a single hop
    paths_exploded = paths_with_id.withColumn("node_in_path", F.explode(F.col("node_id_list")))

    # Join in node attribute information
    exploded_paths_with_node_attributes = paths_exploded.join(
        nodes.select("id", *node_attribute_list).withColumnRenamed("id", "node_in_path"),
        on=["node_in_path"],
        how="inner",
    )

    # Collect node attributes as a dictionary
    exploded_paths_with_node_attributes = exploded_paths_with_node_attributes.withColumn(
        "node_attributes",
        F.create_map(
            *[x for attribute in node_attribute_list for x in [F.lit(attribute), F.col(attribute).cast("string")]],
        ),
    )

    # Collapse node attribute information into a list for each path
    exploded_paths_with_node_attributes = exploded_paths_with_node_attributes.groupBy("path_id").agg(
        F.collect_list("node_attributes").alias("node_attributes_list")
    )

    # Combine with existing columns in original paths dataframe
    paths_with_node_attributes = exploded_paths_with_node_attributes.join(
        paths_with_id, on=["path_id"], how="inner"
    ).drop("path_id")

    return paths_with_node_attributes


def enrich_paths_with_edge_attributes(
    paths: DataFrame,
    edges_processed: DataFrame,
) -> DataFrame:
    """Enrich a paths dataframe with edge attribute information.

    Args:
        paths: Dataframe describing a set of paths containing a "nodes_list" column.
        edges_processed: Preprocessed edges dataframe.

    Returns:
        Dataframe with the same columns as paths, but with additional array-valued columns for each edge attribute.
    """
    # Add unique ID for each path
    paths_with_id = paths.withColumn("path_id", F.monotonically_increasing_id())

    # Explode paths into hops so that each row corresponds to a single hop
    paths_exploded = (
        paths_with_id.withColumns(  # Add lists for the start and end nodes of the hops
            {
                "start_nodes": F.slice(F.col("node_id_list"), 1, F.size("node_id_list") - 1),
                "end_nodes": F.slice(F.col("node_id_list"), 2, F.size("node_id_list") - 1),
            }
        )
        .withColumn(  # Then explode so that hops are represented as pairs
            "hop_pairs", F.explode(F.arrays_zip("start_nodes", "end_nodes"))
        )
        .withColumn(  # Extract hop start and end nodes from the exploded pairs
            "subject", F.col("hop_pairs.start_nodes")
        )
        .withColumn("object", F.col("hop_pairs.end_nodes"))
    )

    # Join in edge attribute information
    exploded_paths_with_attributes = paths_exploded.join(edges_processed, on=["subject", "object"], how="inner")

    # Collapse hop edge attribute information into a list for each path
    exploded_paths_with_attributes = exploded_paths_with_attributes.groupBy("path_id", "start_nodes", "end_nodes").agg(
        F.collect_list("edge_attributes").alias("edge_attributes_list")
    )

    # Combine with existing columns in original paths dataframe
    exploded_paths_with_attributes = (
        exploded_paths_with_attributes.select("path_id", "edge_attributes_list")
        .join(paths_with_id, on=["path_id"], how="inner")
        .drop("path_id")
    )

    return exploded_paths_with_attributes
