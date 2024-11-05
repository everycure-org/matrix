"""Contains functions that generate clauses for constructing complex Neo4j queries."""


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
