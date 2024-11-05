"""Contains functions that generate clauses for constructing complex Neo4j queries."""


def return_clause(limit: int = None) -> str:
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
