import itertools

import pyspark.sql.functions as F
import pytest
from matrix.pipelines.moa.pathfinding import (
    enrich_paths_with_edge_attributes,
    enrich_paths_with_node_attributes,
    get_connecting_paths,
    preprocess_edges,
)
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for all tests"""
    return SparkSession.builder.getOrCreate()


@pytest.fixture
def complete_graph(spark):
    """Create a complete directed graph with self-loops"""
    N_complete = 5
    nodes_data = [(f"id_{i}", f"name_{i}", f"category_{i}") for i in range(N_complete)]
    nodes_complete = spark.createDataFrame(nodes_data, schema=["id", "name", "category"])

    edges_complete_data = [
        (f"id_{i}", f"id_{j}", f"predicate_{i}_{j}") for i, j in itertools.product(range(N_complete), range(N_complete))
    ]
    edges_complete = spark.createDataFrame(edges_complete_data, schema=["subject", "object", "predicate"])

    return nodes_complete, edges_complete


@pytest.fixture
def single_path_graph(spark):
    """Create a graph with a single 4-hop path with first edge flipped"""
    nodes_data = [
        ("source", "source_name", "source_cat"),
        ("a", "a_name", "a_cat"),
        ("b", "b_name", "b_cat"),
        ("c", "c_name", "c_cat"),
        ("target", "target_name", "target_cat"),
    ]
    nodes = spark.createDataFrame(nodes_data, schema=["id", "name", "category"])

    # Edges forming a path source <- a -> b -> c -> target
    edges_data = [
        ("a", "source", "first_edge"),  # Flipped edge
        ("a", "b", "second_edge"),
        ("b", "c", "third_edge"),
        ("c", "target", "fourth_edge"),
    ]
    edges = spark.createDataFrame(edges_data, schema=["subject", "object", "predicate"])

    return nodes, edges


def test_preprocess_edges_complete_graph(spark, complete_graph):
    """Test preprocessing of edges for complete graph"""
    # Given: A complete graph
    _, edges = complete_graph

    # When: Preprocessing the edges
    preprocessed = preprocess_edges(edges)

    # Then: Complete graph already contains all edges in both directions so length is unchanged
    assert preprocessed.count() == edges.count()

    # And: Each edge should have two attributes (forward and backward)
    sample_edge = preprocessed.filter(F.col("subject") == "id_0").filter(F.col("object") == "id_1").first()
    assert len(sample_edge.edge_attributes) == 2

    # And: The attributes should contain directionality information
    directions = set([attr["is_forward"] for attr in sample_edge.edge_attributes])
    assert directions == {"true", "false"}


def test_get_connecting_paths_single_path(spark, single_path_graph):
    """Test path finding on graph with single 4-hop path"""
    # Given: A graph with a single 4-hop path
    nodes, edges = single_path_graph
    preprocessed_edges = preprocess_edges(edges)

    # And: Source and target nodes
    pairs = spark.createDataFrame([("source", "target")], ["source", "target"])

    # When: Finding 2-hop, 3-hop and 4-hop paths
    paths_two_hop = get_connecting_paths(pairs, preprocessed_edges, n_hops=2)
    paths_three_hop = get_connecting_paths(pairs, preprocessed_edges, n_hops=3)
    paths_four_hop = get_connecting_paths(pairs, preprocessed_edges, n_hops=4)

    # Then: Should find exactly one 4-hop path
    assert paths_four_hop.count() == 1

    # And: The 4-hop path should contain the correct sequence of nodes
    path = paths_four_hop.first()
    assert path.node_id_list == ["source", "a", "b", "c", "target"]

    # And: no 3-hop or 2-hop paths should be found
    assert paths_three_hop.count() == 0
    assert paths_two_hop.count() == 0


@pytest.mark.parametrize("k", [2, 3, 4])
def test_get_connecting_paths_complete_graph(spark, complete_graph, k):
    """Test path finding on complete graph"""
    # Given: A complete graph
    _, edges = complete_graph
    preprocessed_edges = preprocess_edges(edges)

    # And: A pair of nodes
    pairs = spark.createDataFrame([("id_0", "id_4")], ["source", "target"])

    # When: Finding k-hop paths
    paths = get_connecting_paths(pairs, preprocessed_edges, n_hops=k, no_repeats=False)

    # Then: Should find N_complete^{k-1} paths (one through each intermediate node)
    N_complete = 5
    assert paths.count() == N_complete ** (k - 1)

    # And: The schema should be correct
    assert len(paths.columns) == 3
    assert "source" in paths.columns and paths.schema["source"].dataType == StringType()
    assert "target" in paths.columns and paths.schema["target"].dataType == StringType()
    assert "node_id_list" in paths.columns and isinstance(paths.schema["node_id_list"].dataType, ArrayType)
    assert paths.schema["node_id_list"].dataType.elementType == StringType()
    assert all(len(row.node_id_list) == k + 1 for row in paths.collect())


def test_enrich_paths_with_node_attributes(spark, single_path_graph):
    """Test path enrichment with node attributes"""
    # Given: A graph with a single path
    nodes, edges = single_path_graph
    preprocessed_edges = preprocess_edges(edges)

    # And: A found path
    pairs = spark.createDataFrame([("source", "target")], ["source", "target"])
    paths = get_connecting_paths(pairs, preprocessed_edges, n_hops=4)

    # When: Enriching with node attributes
    enriched = enrich_paths_with_node_attributes(paths, nodes, ["name", "category"])

    # Then: Should maintain the same number of paths
    assert enriched.count() == 1

    # And: Should have node attributes for each node in the path
    path = enriched.first()
    assert len(path.node_attributes_list) == 5  # 5 nodes in path

    # And: Each node should have name and category attributes
    first_node_attrs = path.node_attributes_list[0]
    assert set(first_node_attrs.keys()) == {"name", "category"}


def test_enrich_paths_with_edge_attributes(spark, single_path_graph):
    """Test path enrichment with edge attributes"""
    # Given: A graph with a single path
    nodes, edges = single_path_graph
    preprocessed_edges = preprocess_edges(edges)

    # And: A found path
    pairs = spark.createDataFrame([("source", "target")], ["source", "target"])
    paths = get_connecting_paths(pairs, preprocessed_edges, n_hops=4)

    # When: Enriching with edge attributes
    enriched = enrich_paths_with_edge_attributes(paths, preprocessed_edges)

    # Then: Should maintain the same number of paths
    assert enriched.count() == 1

    # And: Should have edge attributes for each edge in the path
    path = enriched.first()
    assert len(path.edge_attributes_list) == 4  # 4 edges in path

    # And: Each edge should have predicate and is_forward attributes
    assert all(
        set(edge_attrs.keys()) == {"predicate", "is_forward"}
        for edge_attrs_list in path.edge_attributes_list
        for edge_attrs in edge_attrs_list
    )

    # And: First edge should be marked as backward (is_forward = false)
    directions_list = [edge_attrs_list[0]["is_forward"] for edge_attrs_list in path.edge_attributes_list]
    assert directions_list == ["false", "true", "true", "true"]
