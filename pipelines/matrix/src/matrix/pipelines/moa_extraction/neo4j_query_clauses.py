"""Contains functions that generate clauses for constructing complex Neo4j queries."""

from typing import List


def generate_match_clause(num_hops: int, unidirectional: bool) -> str:
    """Construct an intermediate match clause.

    Example: "-[r1]->(a1)-[r2]->(a2)->[r3]->"

    num_hops: Number of hops in the path.
    unidirectional: Whether to map onto unidirectional paths only.
    """
    edge_end = "->" if unidirectional else "-"

    match_clause_parts = [f"-[r1]{edge_end}"]
    for i in range(1, num_hops):
        match_clause_parts.append(f"(a{i})")
        match_clause_parts.append(f"-[r{i+1}]{edge_end}")
    return "".join(match_clause_parts)


def generate_node_condition_where_clause(num_hops: int, intermediate_ids: List[str]) -> str:
    """Construct the where clause for a path mapping query.

    Example: "(a1.id in ['ID:1', 'ID:2']) AND (a2.id in ['ID:1', 'ID:2'])

    Args:
        num_hops: The number of hops in the path.
        intermediate_ids: The list of intermediate node IDs.
    """
    return " AND ".join([f"(a{i}.id in {str(intermediate_ids)})" for i in range(1, num_hops)])
