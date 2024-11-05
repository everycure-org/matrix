"""Contains functions that generate clauses for constructing complex Neo4j queries."""

from typing import List


def generate_return_clause(limit: int = None) -> str:
    """Generates a RETURN clause of the form expected by KGPaths.add_paths_from_result method.

    Args:
        limit: Maximum number of paths to return. Defaults to None (no limit).
    """
    if limit is None:
        limit_clause = ""
    else:
        limit_clause = f"LIMIT {limit}"

    return f"""WITH path, 
                    nodes(path) as nodes, 
                    relationships(path) as rels
                RETURN [n in nodes | n.name] as node_names,
                    [n in nodes | n.id] as node_ids,
                    [n in nodes | n.category] as node_categories,
                    [r in rels | type(r)] as edge_types,
                    [r in rels | startNode(r) = nodes[apoc.coll.indexOf(rels, r)]] as edge_directions
                    {limit_clause}"""


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


def generate_edge_omission_where_clause(
    edge_omission_rules: dict, num_hops: int, prefix: str = "_moa_extraction_"
) -> str:
    """Construct the where clause enforcing edge omission rules.

    E.g. NONE(r IN relationships(path) WHERE r._moa_extraction_drug_disease) AND (NOT r3._moa_extraction_disease_disease)

    Args:
        edge_omission_rules: The edge omission rules to match.  This takes the form of a dictionary with keys:
            'all', 1, 2, ...
            'all' are the edge tags to omit from all hops
            1, 2, ... are edge tags to omit for the first hop, second hop, ... respectively.
            e.g. edge_omission_rules = {'all': ['drug_disease'], 3: ['disease_disease']}
        num_hops: The number of hops in the paths.
        prefix: The prefix for the tag.
    """
    # Handle 'all' rules if present
    where_clause_parts = []
    for tag in edge_omission_rules["all"]:
        where_clause_parts.append(f"NONE(r IN relationships(path) WHERE r.{prefix}{tag})")

    # Handle hop-specific rules
    for hop in range(1, num_hops + 1):
        hop_rules = edge_omission_rules.get(hop, [])
        for tag in hop_rules:
            where_clause_parts.append(f"(NOT r{hop}.{prefix}{tag})")

    return " AND ".join(where_clause_parts)


def generate_node_condition_where_clause(num_hops: int, intermediate_ids: List[str]) -> str:
    """Construct the where clause for a path mapping query.

    Example: "(a1.id in ['ID:1', 'ID:2']) AND (a2.id in ['ID:1', 'ID:2'])

    Args:
        num_hops: The number of hops in the path.
        intermediate_ids: The list of intermediate node IDs.
    """
    return " AND ".join([f"(a{i}.id in {str(intermediate_ids)})" for i in range(1, num_hops)])
