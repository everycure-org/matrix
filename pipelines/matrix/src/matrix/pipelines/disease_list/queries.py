"""SPARQL query functions for disease list pipeline.

This module breaks down the complex matrix-disease-list-filters.sparql query
into smaller, focused queries that execute faster in PyOxigraph.

Instead of one massive query with 30+ OPTIONAL blocks, we run multiple simple queries
and assemble the results in Python/Pandas. This leverages PyOxigraph's strength
(fast simple queries) while avoiding its weakness (complex query optimization).
"""

import logging
from typing import Set

import pandas as pd

logger = logging.getLogger(__name__)


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
    
    results_df = pd.DataFrame(rows)
    return clean_sparql_results(results_df)

def clean_sparql_results(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = (
            df[col]
            .str.replace("?", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.replace("http://purl.obolibrary.org/obo/MONDO_", "MONDO:", regex=False)
            .str.replace("http://purl.obolibrary.org/obo/mondo#", "mondo:", regex=False)
        )
    df.columns = [col.replace("?", "") for col in df.columns]
    return df

def _extract_ids_from_query(store, query: str, column_name: str = 'category_class') -> Set[str]:
    """Helper function to run a query and extract IDs safely.

    Handles empty DataFrames and missing columns gracefully.

    Args:
        store: PyOxigraph Store object
        query: SPARQL query string
        column_name: Name of column containing IDs (default: 'category_class')

    Returns:
        Set of IDs, or empty set if no results
    """
    df = run_sparql_select(store, query)
    if df.empty or column_name not in df.columns:
        return set()
    return set(df[column_name].tolist())


# =============================================================================
# BASE QUERY
# =============================================================================

def query_base_disease_list(store) -> pd.DataFrame:
    """Get all human diseases with basic metadata (label, definition).

    This is the foundation query that gets all MONDO diseases that are
    descendants of 'human disease' (MONDO:0700096).

    Args:
        store: PyOxigraph Store object

    Returns:
        DataFrame with columns: category_class, label, definition
    """
    query = """
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>

SELECT ?category_class ?label ?definition
WHERE {
  ?category_class rdfs:subClassOf* MONDO:0700096 .

  OPTIONAL {
    ?category_class rdfs:label ?label .
  }

  OPTIONAL {
    ?category_class IAO:0000115 ?definition .
  }

  FILTER(!isBlank(?category_class) && STRSTARTS(str(?category_class), "http://purl.obolibrary.org/obo/MONDO_"))
}
"""
    logger.info("Running base disease list query")
    return run_sparql_select(store, query)


# =============================================================================
# METADATA QUERIES
# =============================================================================

def query_metadata_synonyms(store) -> pd.DataFrame:
    """Get concatenated synonyms for diseases.

    Returns:
        DataFrame with columns: category_class, synonyms
    """
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_synonym; separator="; ") AS ?synonyms)
WHERE {
  ?category_class oio:hasExactSynonym ?synonym .
  BIND(STR(?synonym) AS ?sorted_synonym)
}
GROUP BY ?category_class
ORDER BY ?sorted_synonym
"""
    logger.info("Running synonyms metadata query")
    return run_sparql_select(store, query)


def query_metadata_subsets(store) -> pd.DataFrame:
    """Get concatenated subsets for diseases.

    Returns:
        DataFrame with columns: category_class, subsets
    """
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_subset; separator="; ") AS ?subsets)
WHERE {
  ?category_class oio:inSubset ?subset .
  BIND(STR(?subset) AS ?sorted_subset)
}
GROUP BY ?category_class
ORDER BY ?sorted_subset
"""
    logger.info("Running subsets metadata query")
    return run_sparql_select(store, query)


def query_metadata_crossreferences(store) -> pd.DataFrame:
    """Get concatenated cross-references for diseases.

    Returns:
        DataFrame with columns: category_class, crossreferences
    """
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_xref; separator="; ") AS ?crossreferences)
WHERE {
  ?category_class oio:hasDbXref ?xref .
  BIND(STR(?xref) AS ?sorted_xref)
}
GROUP BY ?category_class
ORDER BY ?sorted_xref
"""
    logger.info("Running cross-references metadata query")
    return run_sparql_select(store, query)


def query_metadata_malacards_linkouts(store) -> pd.DataFrame:
    """Get concatenated MalaCards linkouts for diseases.

    Returns:
        DataFrame with columns: category_class, malacards_linkouts
    """
    query = """
PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_malacardslinkouts; separator="; ") AS ?malacards_linkouts)
WHERE {
  ?category_class mondo:curated_content_resource ?malacards_linkout .
  ?malacards_assertion a owl:Axiom ;
    owl:annotatedSource ?category_class ;
    owl:annotatedProperty mondo:curated_content_resource ;
    owl:annotatedTarget ?malacards_linkout ;
    oio:source "MONDO:MalaCards" .
  FILTER(STRSTARTS(STR(?malacards_linkout), "https://www.malacards.org/"))
  BIND(STR(?malacards_linkout) AS ?sorted_malacardslinkouts)
}
GROUP BY ?category_class
ORDER BY ?sorted_malacardslinkouts
"""
    logger.info("Running MalaCards linkouts metadata query")
    return run_sparql_select(store, query)


# =============================================================================
# FILTER QUERIES - Subset-based
# =============================================================================

def query_filter_matrix_manually_included(store) -> Set[str]:
    """Get diseases manually included in the matrix."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#matrix_included> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_matrix_manually_excluded(store) -> Set[str]:
    """Get diseases manually excluded from the matrix."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#matrix_excluded> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_clingen(store) -> Set[str]:
    """Get diseases curated by ClinGen."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#clingen> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_susceptibility(store) -> Set[str]:
    """Get susceptibility match diseases."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#susceptibility_match> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_mondo_subtype(store) -> Set[str]:
    """Get diseases that are MONDO subtypes."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#mondo_subtype> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_pathway_defect(store) -> Set[str]:
    """Get diseases that are pathway defects."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#pathway_defect> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_grouping_subset(store) -> Set[str]:
    """Get diseases that are designated grouping classes in MONDO."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  VALUES ?_subset {
    <http://purl.obolibrary.org/obo/mondo#ordo_group_of_disorders>
    <http://purl.obolibrary.org/obo/mondo#disease_grouping>
    <http://purl.obolibrary.org/obo/mondo#matrix_grouping>
    <http://purl.obolibrary.org/obo/mondo#harrisons_view>
    <http://purl.obolibrary.org/obo/mondo#rare_grouping>
  }
  ?category_class oio:inSubset ?_subset .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_obsoletion_candidate(store) -> Set[str]:
    """Get diseases that are obsoletion candidates."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#obsoletion_candidate> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_orphanet_subtype(store) -> Set[str]:
    """Get diseases that correspond to Orphanet subtypes."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#ordo_subtype_of_a_disorder> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_orphanet_disorder(store) -> Set[str]:
    """Get diseases that correspond to Orphanet disorders."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#ordo_disorder> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_icd_billable(store) -> Set[str]:
    """Get diseases with billable ICD-10 codes."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?category_class WHERE {
  ?category_class oio:inSubset <http://purl.obolibrary.org/obo/mondo#icd10_billable> .
}
"""
    return _extract_ids_from_query(store, query)


# =============================================================================
# FILTER QUERIES - Hierarchy-based (rdfs:subClassOf*)
# =============================================================================

def query_filter_paraphilic(store) -> Set[str]:
    """Get paraphilic disorders (descendants of MONDO:0000596)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0000596 .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_cardiovascular(store) -> Set[str]:
    """Get cardiovascular disorders (descendants of MONDO:0004995)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0004995 .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_heart_disorder(store) -> Set[str]:
    """Get heart disorders (descendants of MONDO:0005267)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0005267 .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_inflammatory(store) -> Set[str]:
    """Get inflammatory diseases (descendants of MONDO:0021166)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0021166 .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_psychiatric(store) -> Set[str]:
    """Get psychiatric disorders (descendants of MONDO:0002025)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0002025 .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_cancer_or_benign_tumor(store) -> Set[str]:
    """Get cancer or benign tumor (descendants of MONDO:0045024)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0045024 .
}
"""
    return _extract_ids_from_query(store, query)


# =============================================================================
# FILTER QUERIES - Label-based
# =============================================================================

def query_filter_withorwithout(store) -> Set[str]:
    """Get diseases with 'with or without' in the label."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?category_class WHERE {
  ?category_class rdfs:label ?label .
  FILTER(CONTAINS(?label, "with or without"))
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_andor(store) -> Set[str]:
    """Get diseases with 'and/or' in the label."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?category_class WHERE {
  ?category_class rdfs:label ?label .
  FILTER(CONTAINS(?label, "and/or"))
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_acquired(store) -> Set[str]:
    """Get diseases starting with 'acquired' in the label."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?category_class WHERE {
  ?category_class rdfs:label ?label .
  FILTER(STRSTARTS(?label, "acquired "))
}
"""
    return _extract_ids_from_query(store, query)


# =============================================================================
# FILTER QUERIES - Complex
# =============================================================================

def query_filter_unclassified_hereditary(store) -> Set[str]:
    """Get unclassified hereditary diseases (leaf nodes under hereditary disease with no other parents)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

SELECT ?entity WHERE {
  ?entity rdfs:subClassOf+ MONDO:0003847 .

  FILTER NOT EXISTS {
    # is a leaf (does not have children)
    ?x rdfs:subClassOf ?entity .
    FILTER(?x != ?entity)
  }

  FILTER NOT EXISTS {
    # does not have any other parent other than hereditary disease
    ?entity rdfs:subClassOf ?y .
    FILTER(
      (?y != MONDO:0003847)
      && (?y != MONDO:0000001)
      && (?y != MONDO:0700096)
      && (?y != MONDO:0008577)
      && (?entity != ?y))
  }
}
"""
    # Note: query uses ?entity instead of ?category_class
    return _extract_ids_from_query(store, query, column_name='entity')


def query_filter_grouping_subset_ancestor(store) -> Set[str]:
    """Get ancestors of designated grouping classes."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT DISTINCT ?category_class WHERE {
  ?grouping_class_subset rdfs:subClassOf+ ?category_class .
  VALUES ?_subset {
    <http://purl.obolibrary.org/obo/mondo#ordo_group_of_disorders>
    <http://purl.obolibrary.org/obo/mondo#disease_grouping>
    <http://purl.obolibrary.org/obo/mondo#harrisons_view>
    <http://purl.obolibrary.org/obo/mondo#rare_grouping>
  }
  ?grouping_class_subset oio:inSubset ?_subset .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_orphanet_subtype_descendant(store) -> Set[str]:
    """Get descendants of Orphanet subtypes."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

SELECT DISTINCT ?category_class WHERE {
  ?category_class rdfs:subClassOf+ ?subtype_subset .
  ?subtype_subset oio:inSubset <http://purl.obolibrary.org/obo/mondo#ordo_subtype_of_a_disorder> .
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_omimps(store) -> Set[str]:
    """Get diseases corresponding to OMIM Phenotypic Series."""
    query = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?category_class WHERE {
  ?category_class skos:exactMatch ?match .
  FILTER(STRSTARTS(STR(?match), "https://omim.org/phenotypicSeries/PS"))
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_omimps_descendant(store) -> Set[str]:
    """Get descendants of OMIM Phenotypic Series."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?category_class WHERE {
  ?category_class rdfs:subClassOf+ ?omimps .
  ?omimps skos:exactMatch ?match .
  FILTER(STRSTARTS(STR(?match), "https://omim.org/phenotypicSeries/PS"))
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_omim(store) -> Set[str]:
    """Get diseases with OMIM entries."""
    query = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?category_class WHERE {
  ?category_class skos:exactMatch ?match .
  FILTER(STRSTARTS(STR(?match), "https://omim.org/entry/"))
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_leaf(store) -> Set[str]:
    """Get leaf nodes (diseases with no children)."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?category_class WHERE {
  ?category_class rdf:type owl:Class .
  FILTER NOT EXISTS {
    ?leaf_x rdfs:subClassOf ?category_class
  }
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_leaf_direct_parent(store) -> Set[str]:
    """Get direct parents of leaf nodes."""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?category_class WHERE {
  ?leaf rdfs:subClassOf ?category_class .
  FILTER NOT EXISTS {
    ?leaf_direct_parent_x rdfs:subClassOf ?leaf
  }
}
"""
    return _extract_ids_from_query(store, query)


# =============================================================================
# FILTER QUERIES - ICD-10 based
# =============================================================================

def query_filter_icd_category(store) -> Set[str]:
    """Get diseases corresponding to ICD-10 categories (has . but not -)."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?category_class WHERE {
  ?category_class oio:hasDbXref ?xref .
  ?a owl:annotatedSource ?category_class ;
     owl:annotatedProperty oio:hasDbXref ;
     owl:annotatedTarget ?xref ;
     oio:source ?type
  FILTER (
      STRSTARTS(str(?xref), "ICD10") &&
      !CONTAINS(SUBSTR(str(?xref), 6), "-") &&
      CONTAINS(SUBSTR(str(?xref), 6), ".")
  )
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_icd_chapter_code(store) -> Set[str]:
    """Get diseases corresponding to ICD-10 chapter codes (has - but not .)."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?category_class WHERE {
  ?category_class oio:hasDbXref ?xref .
  ?a owl:annotatedSource ?category_class ;
     owl:annotatedProperty oio:hasDbXref ;
     owl:annotatedTarget ?xref ;
     oio:source ?type
  FILTER (
      STRSTARTS(str(?xref), "ICD10") &&
      CONTAINS(SUBSTR(str(?xref), 6), "-") &&
      !CONTAINS(SUBSTR(str(?xref), 6), ".")
  )
}
"""
    return _extract_ids_from_query(store, query)


def query_filter_icd_chapter_header(store) -> Set[str]:
    """Get diseases corresponding to ICD-10 chapter headers (no . or -)."""
    query = """
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?category_class WHERE {
  ?category_class oio:hasDbXref ?xref .
  ?a owl:annotatedSource ?category_class ;
     owl:annotatedProperty oio:hasDbXref ;
     owl:annotatedTarget ?xref ;
     oio:source ?type
  FILTER (
      STRSTARTS(str(?xref), "ICD10") &&
      !CONTAINS(SUBSTR(str(?xref), 6), "-") &&
      !CONTAINS(SUBSTR(str(?xref), 6), ".")
  )
}
"""
    return _extract_ids_from_query(store, query)


# =============================================================================
# ASSEMBLY FUNCTION
# =============================================================================

def assemble_disease_list(store) -> pd.DataFrame:
    """Assemble complete disease list with all metadata and filters.

    This is the main orchestrator function that runs all queries and assembles
    the results into a single DataFrame matching the schema of the original
    mega-query.

    Args:
        store: PyOxigraph Store object

    Returns:
        DataFrame with all disease list columns
    """
    logger.info("Assembling disease list from multiple focused queries")

    # 1. Get base disease list
    df = query_base_disease_list(store)
    logger.info(f"Base query returned {len(df)} diseases")

    # 2. Add metadata (left joins)
    logger.info("Adding metadata...")

    synonyms = query_metadata_synonyms(store)
    df = df.merge(synonyms, on='category_class', how='left')

    subsets = query_metadata_subsets(store)
    df = df.merge(subsets, on='category_class', how='left')

    crossrefs = query_metadata_crossreferences(store)
    df = df.merge(crossrefs, on='category_class', how='left')

    malacards = query_metadata_malacards_linkouts(store)
    df = df.merge(malacards, on='category_class', how='left')

    # 3. Add filter flags
    logger.info("Adding filter flags...")

    # Subset-based filters
    df['f_matrix_manually_included'] = df['category_class'].isin(
        query_filter_matrix_manually_included(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_matrix_manually_excluded'] = df['category_class'].isin(
        query_filter_matrix_manually_excluded(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_clingen'] = df['category_class'].isin(
        query_filter_clingen(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_susceptibility'] = df['category_class'].isin(
        query_filter_susceptibility(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_mondo_subtype'] = df['category_class'].isin(
        query_filter_mondo_subtype(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_pathway_defect'] = df['category_class'].isin(
        query_filter_pathway_defect(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_grouping_subset'] = df['category_class'].isin(
        query_filter_grouping_subset(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_obsoletion_candidate'] = df['category_class'].isin(
        query_filter_obsoletion_candidate(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_orphanet_subtype'] = df['category_class'].isin(
        query_filter_orphanet_subtype(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_orphanet_disorder'] = df['category_class'].isin(
        query_filter_orphanet_disorder(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_icd_billable'] = df['category_class'].isin(
        query_filter_icd_billable(store)
    ).apply(lambda x: "TRUE" if x else "")

    # Hierarchy-based filters
    df['f_paraphilic'] = df['category_class'].isin(
        query_filter_paraphilic(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_cardiovascular'] = df['category_class'].isin(
        query_filter_cardiovascular(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_filter_heart_disorder'] = df['category_class'].isin(
        query_filter_heart_disorder(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_inflammatory'] = df['category_class'].isin(
        query_filter_inflammatory(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_psychiatric'] = df['category_class'].isin(
        query_filter_psychiatric(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_cancer_or_benign_tumor'] = df['category_class'].isin(
        query_filter_cancer_or_benign_tumor(store)
    ).apply(lambda x: "TRUE" if x else "")

    # Label-based filters
    df['f_withorwithout'] = df['category_class'].isin(
        query_filter_withorwithout(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_andor'] = df['category_class'].isin(
        query_filter_andor(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_acquired'] = df['category_class'].isin(
        query_filter_acquired(store)
    ).apply(lambda x: "TRUE" if x else "")

    # Complex filters
    df['f_unclassified_hereditary'] = df['category_class'].isin(
        query_filter_unclassified_hereditary(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_grouping_subset_ancestor'] = df['category_class'].isin(
        query_filter_grouping_subset_ancestor(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_orphanet_subtype_descendant'] = df['category_class'].isin(
        query_filter_orphanet_subtype_descendant(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_omimps'] = df['category_class'].isin(
        query_filter_omimps(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_omimps_descendant'] = df['category_class'].isin(
        query_filter_omimps_descendant(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_omim'] = df['category_class'].isin(
        query_filter_omim(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_leaf'] = df['category_class'].isin(
        query_filter_leaf(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_leaf_direct_parent'] = df['category_class'].isin(
        query_filter_leaf_direct_parent(store)
    ).apply(lambda x: "TRUE" if x else "")

    # ICD-10 filters
    df['f_icd_category'] = df['category_class'].isin(
        query_filter_icd_category(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_icd_chapter_code'] = df['category_class'].isin(
        query_filter_icd_chapter_code(store)
    ).apply(lambda x: "TRUE" if x else "")

    df['f_icd_chapter_header'] = df['category_class'].isin(
        query_filter_icd_chapter_header(store)
    ).apply(lambda x: "TRUE" if x else "")

    # Sort by label descending (matching original query)
    df = df.sort_values('label', ascending=False)

    logger.info(f"Assembled disease list with {len(df)} diseases and {len(df.columns)} columns")

    return df

def query_get_ancestors(store, child_id: str) -> set[str]:
    """Get all ancestors (transitive parents) of a node using SPARQL.

    Args:
        store: PyOxigraph Store object
        child_id: Child node ID (e.g., "MONDO:0000001")

    Returns:
        Set of ancestor IDs in CURIE format
    """
    # Convert CURIE to URI for SPARQL
    uri = child_id.replace("MONDO:", "http://purl.obolibrary.org/obo/MONDO_")

    # SPARQL query to find all ancestors using property paths
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX obo: <http://purl.obolibrary.org/obo/>

        SELECT DISTINCT ?ancestor WHERE {{
            <{uri}> rdfs:subClassOf* ?ancestor .
            FILTER(?ancestor != <{uri}>)
        }}
    """

    results = store.query(query)
    ancestors = set()
    for result in results:
        anc_uri = str(result['ancestor'])
        # Convert URI back to CURIE
        if 'MONDO_' in anc_uri:
            curie = anc_uri.replace("http://purl.obolibrary.org/obo/MONDO_", "MONDO:")
            ancestors.add(curie)

    return ancestors


def query_get_descendants(store, root_id: str) -> set[str]:
    """Get all descendants (transitive children) of a node using SPARQL.

    Args:
        store: PyOxigraph Store object
        root_id: Root node ID (e.g., "MONDO:0000001")

    Returns:
        Set of descendant IDs in CURIE format
    """
    # Convert CURIE to URI for SPARQL
    uri = root_id.replace("MONDO:", "http://purl.obolibrary.org/obo/MONDO_")

    # SPARQL query to find all descendants using property paths
    # rdfs:subClassOf* means zero or more subClassOf relationships
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX obo: <http://purl.obolibrary.org/obo/>

        SELECT DISTINCT ?descendant WHERE {{
            ?descendant rdfs:subClassOf* <{uri}> .
            FILTER(?descendant != <{uri}>)
        }}
    """

    results = store.query(query)
    descendants = set()
    for result in results:
        desc_uri = str(result['descendant'])
        # Convert URI back to CURIE
        if 'MONDO_' in desc_uri:
            curie = desc_uri.replace("http://purl.obolibrary.org/obo/MONDO_", "MONDO:")
            descendants.add(curie)

    return descendants


# =============================================================================
# Additional Queries (originally from separate .sparql/.ru files)
# =============================================================================


def query_mondo_labels() -> str:
    """Get MONDO labels query string.

    Returns SPARQL query to extract all MONDO labels.
    """
    return """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT ?ID ?LABEL

WHERE {
  ?ID rdfs:label ?LABEL .
  FILTER(STRSTARTS(STR(?ID), "http://purl.obolibrary.org/obo/MONDO_"))
}
"""


def query_ontology_metadata() -> str:
    """Get ontology metadata query string.

    Returns SPARQL query to extract ontology version and metadata.
    """
    return """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT ?versionIRI ?IRI ?title

WHERE {
  ?IRI a owl:Ontology .
  OPTIONAL { ?IRI owl:versionIRI ?versionIRI . }
  OPTIONAL { ?IRI dc:title ?title . }
}
"""


def query_mondo_obsoletes() -> str:
    """Get MONDO obsolete terms query string.

    Returns SPARQL query to find all obsolete MONDO terms with replacements.
    """
    return """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>

SELECT
    ?cls
    ?label
    ?deprecated
    (GROUP_CONCAT(DISTINCT STR(?repl); separator="|") AS ?replacements)
    (GROUP_CONCAT(DISTINCT STR(?cons); separator="|") AS ?considers)
WHERE {
    ?cls a owl:Class .
    FILTER(STRSTARTS(STR(?cls), "http://purl.obolibrary.org/obo/MONDO_"))

    OPTIONAL { ?cls rdfs:label ?label. }

    # common OBO annotation properties for replacements / considers
    OPTIONAL { ?cls IAO:0100001 ?repl. }
    OPTIONAL { ?cls oboInOwl:consider ?cons. }

    # detect obsoletion via owl:deprecated = true (robust to literal forms)
    OPTIONAL { ?cls owl:deprecated ?deprecated. }
    FILTER(BOUND(?deprecated) &&
                 ( ?deprecated = true
                     || STR(?deprecated) = "true"
                     || ?deprecated = "true"^^xsd:boolean ))
}
GROUP BY ?cls ?label ?deprecated
ORDER BY ?cls
"""


def query_matrix_disease_list_metrics() -> str:
    """Get disease metrics query string.

    Returns SPARQL query to count descendants for each disease.
    """
    return """
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX def: <http://purl.obolibrary.org/obo/IAO_0000115>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT
  ?category_class
  (COUNT(DISTINCT ?decendant_disease) AS ?count_descendants)
WHERE
{
  # We are only looking for classes that are specifically human diseases.
  ?category_class rdfs:subClassOf* MONDO:0700096 .

  ########################
  #### Metadata ##########
  ########################

  OPTIONAL {
    ?decendant_disease rdfs:subClassOf+ ?category_class .
  }

  FILTER( !isBlank(?category_class) && STRSTARTS(str(?category_class), "http://purl.obolibrary.org/obo/MONDO_"))
}
GROUP BY ?category_class
ORDER BY DESC(?category_class)
"""


def query_inject_mondo_top_grouping() -> str:
    """Get SPARQL UPDATE query to inject mondo_top_grouping subset.

    Returns SPARQL UPDATE query to add top-level grouping subset annotation.
    """
    return """
PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

INSERT {
  ?subject oio:inSubset <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping> .
}
WHERE {
  ?subject rdfs:subClassOf <http://purl.obolibrary.org/obo/MONDO_0700096> .
}
"""


def query_inject_susceptibility_subset() -> str:
    """Get SPARQL UPDATE query to inject susceptibility subset annotations.

    Returns SPARQL UPDATE query to mark susceptibility diseases.
    """
    return """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX pattern: <http://purl.obolibrary.org/obo/mondo/patterns/>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

INSERT {
    ?subset rdfs:subPropertyOf oboInOwl:SubsetProperty .
    ?entity oboInOwl:inSubset ?subset .
}

WHERE {
{
    ?entity rdfs:subClassOf+ MONDO:0042489 .
    BIND(IRI(CONCAT("http://purl.obolibrary.org/obo/mondo#","susceptibility_mondo")) AS ?subset)
} UNION {
    ?entity rdfs:subClassOf+ MONDO:0000001 .
    ?entity rdfs:label ?label .
    FILTER(regex(str(?label), "susceptib") || regex(str(?label), "predisposit") )
    BIND(IRI(CONCAT("http://purl.obolibrary.org/obo/mondo#","susceptibility_match")) AS ?subset)
}

}
"""


def query_inject_subset_declaration() -> str:
    """Get SPARQL UPDATE query to declare all subsets as SubsetProperty.

    Returns SPARQL UPDATE query to properly declare subset properties.
    """
    return """
PREFIX : <http://www.test.com/ns/test#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

INSERT { ?y rdfs:subPropertyOf <http://www.geneontology.org/formats/oboInOwl#SubsetProperty> . }

WHERE {
  ?x <http://www.geneontology.org/formats/oboInOwl#inSubset>  ?y .
  FILTER(isIRI(?y))
  FILTER(regex(str(?y),"^(http://purl.obolibrary.org/obo/)") || regex(str(?y),"^(http://www.ebi.ac.uk/efo/)") || regex(str(?y),"^(https://w3id.org/biolink/)") || regex(str(?y),"^(http://purl.obolibrary.org/obo)"))
}
"""


def query_downfill_disease_groupings() -> str:
    """Get SPARQL UPDATE query to propagate groupings to descendants.

    Returns SPARQL UPDATE query to downfill subset annotations from parents.
    """
    return """
PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

INSERT {
  ?descendant oio:inSubset ?memberSubset .
  ?descendant oio:inSubset ?subjectSubset .
}
WHERE {
    VALUES ?subset {
      <http://purl.obolibrary.org/obo/mondo#mondo_txgnn>
      <http://purl.obolibrary.org/obo/mondo#harrisons_view>
      <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping>
      }
    ?subject oio:inSubset ?subset ; rdfs:label ?label .
    ?descendant rdfs:subClassOf+ ?subject .
    BIND(IRI(CONCAT(STR(?subset),"_member")) AS ?memberSubset)
    BIND(IRI(CONCAT(CONCAT(STR(?subset),"_"),REPLACE(LCASE(?label), "[^a-zA-Z0-9]", "_"))) AS ?subjectSubset)
}
"""


def query_disease_groupings_other() -> str:
    """Get SPARQL UPDATE query to mark ungrouped diseases as 'other'.

    Returns SPARQL UPDATE query to add 'other' grouping for diseases not in any grouping.
    """
    return """
PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

INSERT {
  ?subject oio:inSubset ?otherSubset .
}
WHERE {
  VALUES ?subset {
    <http://purl.obolibrary.org/obo/mondo#mondo_txgnn>
    <http://purl.obolibrary.org/obo/mondo#harrisons_view>
    <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping>
    <http://purl.obolibrary.org/obo/mondo#txgnn>
    <http://purl.obolibrary.org/obo/mondo#anatomical>
    <http://purl.obolibrary.org/obo/mondo#medical_specialization>
    <http://purl.obolibrary.org/obo/mondo#is_pathogen_caused>
    <http://purl.obolibrary.org/obo/mondo#is_cancer>
    <http://purl.obolibrary.org/obo/mondo#is_glucose_dysfunction>
    <http://purl.obolibrary.org/obo/mondo#tag_existing_treatment>
    <http://purl.obolibrary.org/obo/mondo#tag_qualy_lost>
  }
  ?subject rdfs:subClassOf+ <http://purl.obolibrary.org/obo/MONDO_0700096> .

  # Bind ?memberSubset before checking filters
  BIND(IRI(CONCAT(STR(?subset), "_member")) AS ?memberSubset)
  BIND(IRI(CONCAT(STR(?subset), "_other")) AS ?otherSubset)

  # Ensure ?subject is not already inSubset for ?subset or ?memberSubset
  FILTER NOT EXISTS {
    ?subject oio:inSubset ?subset .
  }
  FILTER NOT EXISTS {
    ?subject oio:inSubset ?memberSubset .
  }
}
"""
