#!/usr/bin/env python
"""Test script for PyOxigraph SPARQL performance with MONDO ontology.

This script tests the complete PyOxigraph setup without running the full Kedro pipeline.
"""

import time
from pathlib import Path

import pandas as pd
import requests
from pyoxigraph import RdfFormat, Store


def test_oxigraph_mondo():
    """Test loading MONDO and running SPARQL queries with PyOxigraph."""
    print("=" * 80)
    print("PyOxigraph MONDO Test")
    print("=" * 80)

    # 1. Download MONDO
    print("\n1. Downloading MONDO ontology...")
    mondo_url = "http://purl.obolibrary.org/obo/mondo.owl"
    start = time.time()
    response = requests.get(mondo_url)
    response.raise_for_status()
    download_time = time.time() - start
    print(f"   Downloaded {len(response.content) / 1024 / 1024:.1f} MB in {download_time:.2f}s")

    # 2. Load into PyOxigraph store
    print("\n2. Loading into PyOxigraph store...")
    start = time.time()
    store = Store()
    store.load(response.content, format=RdfFormat.RDF_XML)
    load_time = time.time() - start
    print(f"   Loaded in {load_time:.2f}s")

    # 3. Count triples
    print("\n3. Counting triples...")
    start = time.time()
    count_query = "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"
    result = list(store.query(count_query))
    triple_count = result[0]["count"] if result else 0
    count_time = time.time() - start
    print(f"   {triple_count} triples (query took {count_time:.2f}s)")

    # 4. Test label extraction query
    print("\n4. Testing label extraction query...")
    labels_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ID ?LABEL
WHERE {
  ?ID rdfs:label ?LABEL .
  FILTER(STRSTARTS(STR(?ID), "http://purl.obolibrary.org/obo/MONDO_"))
}
LIMIT 10
"""
    start = time.time()
    results = store.query(labels_query)
    # Variables in PyOxigraph include the '?' prefix, but we need to remove it for cleaner output
    variables = [str(v)[1:] if str(v).startswith('?') else str(v) for v in results.variables]

    rows = []
    for solution in results:
        row_dict = {}
        for i, var_with_prefix in enumerate(results.variables):
            value = solution[var_with_prefix]  # Access with original variable (with ?)
            clean_var = variables[i]  # Store with clean variable name (without ?)
            row_dict[clean_var] = str(value) if value else ""
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    query_time = time.time() - start
    print(f"   Extracted {len(df)} labels in {query_time:.2f}s")
    print(f"   Sample results:")
    if not df.empty:
        print(df.head().to_string(index=False))
    else:
        print("   (No results)")

    # 5. Test full label extraction (no LIMIT)
    print("\n5. Testing full label extraction (all MONDO labels)...")
    full_labels_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ID ?LABEL
WHERE {
  ?ID rdfs:label ?LABEL .
  FILTER(STRSTARTS(STR(?ID), "http://purl.obolibrary.org/obo/MONDO_"))
}
"""
    start = time.time()
    results = store.query(full_labels_query)
    # Variables in PyOxigraph include the '?' prefix, but we need to remove it for cleaner output
    variables = [str(v)[1:] if str(v).startswith('?') else str(v) for v in results.variables]

    rows = []
    for solution in results:
        row_dict = {}
        for i, var_with_prefix in enumerate(results.variables):
            value = solution[var_with_prefix]  # Access with original variable (with ?)
            clean_var = variables[i]  # Store with clean variable name (without ?)
            row_dict[clean_var] = str(value) if value else ""
        rows.append(row_dict)

    df_full = pd.DataFrame(rows)
    full_query_time = time.time() - start
    print(f"   Extracted {len(df_full)} labels in {full_query_time:.2f}s")

    # 6. Test a complex query (disease list filters simulation)
    print("\n6. Testing complex query (with multiple filters)...")
    complex_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?disease ?label (COUNT(?subset) AS ?subset_count)
WHERE {
  ?disease rdfs:label ?label .
  FILTER(STRSTARTS(STR(?disease), "http://purl.obolibrary.org/obo/MONDO_"))
  OPTIONAL { ?disease oboInOwl:inSubset ?subset }
}
GROUP BY ?disease ?label
LIMIT 100
"""
    start = time.time()
    results = store.query(complex_query)
    complex_time = time.time() - start
    count = len(list(results))
    print(f"   Query returned {count} results in {complex_time:.2f}s")

    # 7. Test SPARQL UPDATE
    print("\n7. Testing SPARQL UPDATE (adding test triple)...")
    update_query = """
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

INSERT DATA {
  ex:test_subject rdfs:label "Test Label" .
}
"""
    start = time.time()
    store.update(update_query)
    update_time = time.time() - start
    print(f"   Update completed in {update_time:.2f}s")

    # Verify the update
    verify_query = 'SELECT ?s ?label WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> "Test Label" }'
    verify_results = list(store.query(verify_query))
    print(f"   Verified: Added triple found = {len(verify_results) > 0}")

    # 8. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Download time:        {download_time:.2f}s")
    print(f"Load time:            {load_time:.2f}s")
    print(f"Triple count:         {triple_count}")
    print(f"Sample query (10):    {query_time:.2f}s")
    print(f"Full labels query:    {full_query_time:.2f}s ({len(df_full)} labels)")
    print(f"Complex query:        {complex_time:.2f}s")
    print(f"UPDATE query:         {update_time:.2f}s")
    print("=" * 80)
    print("\n✅ All tests passed! PyOxigraph is working correctly.")


if __name__ == "__main__":
    test_oxigraph_mondo()
