"""Tests for the Neo4j query clauses."""

from matrix.pipelines.moa_extraction.neo4j_query_clauses import (
    generate_match_clause,
    generate_node_condition_where_clause,
)


def test_generate_match_clause():
    # Given a number of hops and a direction
    # When we generate the match clause
    match_clause_1 = generate_match_clause(3, unidirectional=True)
    match_clause_2 = generate_match_clause(3, unidirectional=False)

    # Then the result is correct
    assert match_clause_1 == "-[r1]->(a1)-[r2]->(a2)-[r3]->"
    assert match_clause_2 == "-[r1]-(a1)-[r2]-(a2)-[r3]-"


def test_generate_node_condition_where_clause():
    # Given a list of intermediate node IDs
    intermediate_ids = ["ID:1", "ID:2"]

    # When we generate the where clause for a 2-hop path
    where_clause = generate_node_condition_where_clause(num_hops=3, intermediate_ids=intermediate_ids)

    # Then the result is correct
    assert where_clause == "(a1.id in ['ID:1', 'ID:2']) AND (a2.id in ['ID:1', 'ID:2'])"
