"""Utilities for working with RDF/OWL ontologies using RDFLIB.

This module provides functions for loading, querying, and manipulating
ontologies using RDFLIB instead of ROBOT.

Uses PyOxigraph directly for significantly faster SPARQL query performance.
"""

from pathlib import Path

import pandas as pd


def load_owl_to_graph(owl_content: str):
    """Load OWL content string into PyOxigraph store.

    Args:
        owl_content: String containing OWL/RDF-XML content

    Returns:
        PyOxigraph Store object for fast SPARQL queries
    """
    import pyoxigraph
    from pyoxigraph import RdfFormat

    # Use PyOxigraph directly for much faster SPARQL
    store = pyoxigraph.Store()
    store.load(owl_content.encode('utf-8'), format=RdfFormat.RDF_XML)
    return store


def run_sparql_select(store, query: str) -> pd.DataFrame:
    """Execute a SPARQL SELECT query using PyOxigraph for fast performance.

    Args:
        store: PyOxigraph Store object
        query: SPARQL query string

    Returns:
        DataFrame with query results (columns match SELECT variables)
    """
    # Execute query using PyOxigraph (much faster than RDFLIB)
    results = store.query(query)

    # Get variable names from the query results (.variables is a property, not a method)
    # Variables in PyOxigraph include the '?' prefix, remove it for cleaner column names
    variables = [str(v)[1:] if str(v).startswith('?') else str(v) for v in results.variables]

    # Convert to DataFrame
    rows = []
    for solution in results:
        row_dict = {}
        for i, var_with_prefix in enumerate(results.variables):
            value = solution[var_with_prefix]  # Access with original variable (with ?)
            clean_var = variables[i]  # Store with clean variable name (without ?)
            # Convert to string, handle None values
            row_dict[clean_var] = str(value) if value else ""
        rows.append(row_dict)

    return pd.DataFrame(rows)

def serialize_store(store, format: str = "xml") -> str:
    """Serialize PyOxigraph Store back to string format.

    Args:
        store: PyOxigraph Store to serialize
        format: Output format (default: "xml" for OWL/RDF-XML)

    Returns:
        Serialized graph as string
    """
    from pyoxigraph import RdfFormat

    # Map format string to PyOxigraph RdfFormat
    format_map = {
        "xml": RdfFormat.RDF_XML,
        "ttl": RdfFormat.TURTLE,
        "nt": RdfFormat.N_TRIPLES,
        "n3": RdfFormat.N3,
    }
    rdf_format = format_map.get(format, RdfFormat.RDF_XML)
    return store.dump(format=rdf_format).decode('utf-8')
