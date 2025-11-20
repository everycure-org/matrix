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
from pyoxigraph import Store

logger = logging.getLogger(__name__)


def _get_sparql_prefixes() -> str:
    """Get common SPARQL prefixes used across all queries."""
    return """PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
"""


def _run_sparql_select(store: Store, query: str) -> pd.DataFrame:
    """Execute a SPARQL SELECT query using PyOxigraph for fast performance."""
    results = store.query(query)

    variables = [str(v)[1:] if str(v).startswith("?") else str(v) for v in results.variables]

    rows = []
    for solution in results:
        row_dict = {}
        for i, var_with_prefix in enumerate(results.variables):
            value = solution[var_with_prefix]
            clean_var = variables[i]
            if value is None:
                row_dict[clean_var] = ""
            elif hasattr(value, "value"):
                row_dict[clean_var] = str(value.value)
            else:
                row_dict[clean_var] = str(value)
        rows.append(row_dict)

    results_df = pd.DataFrame(rows)
    return _clean_sparql_results(results_df)


def _clean_sparql_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean SPARQL result strings by removing prefixes and special characters."""
    for col in df.columns:
        df[col] = (
            df[col]
            .str.replace("?", "", regex=False)
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.replace("http://purl.obolibrary.org/obo/MONDO_", "MONDO:", regex=False)
            .str.replace("http://purl.obolibrary.org/obo/mondo#", "mondo:", regex=False)
        )
    df.columns = [str(col).replace("?", "") for col in df.columns]
    return df


def _query_and_extract_ids_from_column(
    store: Store, query: str, column_name: str = "category_class", mondo_only: bool = True
) -> Set[str]:
    """Run a query and extract IDs safely, handling empty DataFrames and missing columns."""
    df = _run_sparql_select(store, query)
    if df.empty or column_name not in df.columns:
        return set()

    ids = set(df[column_name].tolist())

    if mondo_only:
        ids = {id for id in ids if id.startswith("MONDO:")}

    return ids


# =============================================================================
# METADATA QUERIES
# =============================================================================


def _query_base_disease_list(store: Store) -> pd.DataFrame:
    """Get all human diseases with basic metadata (label, definition)."""
    query = (
        _get_sparql_prefixes()
        + """
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
    )
    logger.info("Running base disease list query")
    return _run_sparql_select(store, query)


def _query_metadata_synonyms(store: Store) -> pd.DataFrame:
    """Get concatenated synonyms for diseases."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_synonym; separator="; ") AS ?synonyms)
WHERE {
  ?category_class oboInOwl:hasExactSynonym ?synonym .
  BIND(STR(?synonym) AS ?sorted_synonym)
}
GROUP BY ?category_class
ORDER BY ?sorted_synonym
"""
    )
    logger.info("Running synonyms metadata query")
    return _run_sparql_select(store, query)


def _query_metadata_subsets(store: Store) -> pd.DataFrame:
    """Get concatenated subsets for diseases."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_subset; separator="; ") AS ?subsets)
WHERE {
  ?category_class oboInOwl:inSubset ?subset .
  BIND(STR(?subset) AS ?sorted_subset)
}
GROUP BY ?category_class
ORDER BY ?sorted_subset
"""
    )
    logger.info("Running subsets metadata query")
    return _run_sparql_select(store, query)


def _query_metadata_crossreferences(store: Store) -> pd.DataFrame:
    """Get concatenated cross-references for diseases."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_xref; separator="; ") AS ?crossreferences)
WHERE {
  ?category_class oboInOwl:hasDbXref ?xref .
  BIND(STR(?xref) AS ?sorted_xref)
}
GROUP BY ?category_class
ORDER BY ?sorted_xref
"""
    )
    logger.info("Running cross-references metadata query")
    return _run_sparql_select(store, query)


def _query_metadata_malacards_linkouts(store: Store) -> pd.DataFrame:
    """Get concatenated MalaCards linkouts for diseases."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class (GROUP_CONCAT(DISTINCT ?sorted_malacardslinkouts; separator="; ") AS ?malacards_linkouts)
WHERE {
  ?category_class mondo:curated_content_resource ?malacards_linkout .
  ?malacards_assertion a owl:Axiom ;
    owl:annotatedSource ?category_class ;
    owl:annotatedProperty mondo:curated_content_resource ;
    owl:annotatedTarget ?malacards_linkout ;
    oboInOwl:source "MONDO:MalaCards" .
  FILTER(STRSTARTS(STR(?malacards_linkout), "https://www.malacards.org/"))
  BIND(STR(?malacards_linkout) AS ?sorted_malacardslinkouts)
}
GROUP BY ?category_class
ORDER BY ?sorted_malacardslinkouts
"""
    )
    logger.info("Running MalaCards linkouts metadata query")
    return _run_sparql_select(store, query)


# =============================================================================
# FILTER QUERIES - Subset-based
# =============================================================================


def _query_filter_matrix_manually_included(store: Store) -> Set[str]:
    """Get diseases manually included in the matrix."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#matrix_included> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_matrix_manually_excluded(store: Store) -> Set[str]:
    """Get diseases manually excluded from the matrix."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#matrix_excluded> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_clingen(store: Store) -> Set[str]:
    """Get diseases curated by ClinGen."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#clingen> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_susceptibility(store: Store) -> Set[str]:
    """Get susceptibility match diseases."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#susceptibility_match> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_mondo_subtype(store: Store) -> Set[str]:
    """Get diseases that are MONDO subtypes."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#mondo_subtype> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_pathway_defect(store: Store) -> Set[str]:
    """Get diseases that are pathway defects."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#pathway_defect> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_grouping_subset(store: Store) -> Set[str]:
    """Get diseases that are designated grouping classes in MONDO."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  VALUES ?_subset {
    <http://purl.obolibrary.org/obo/mondo#ordo_group_of_disorders>
    <http://purl.obolibrary.org/obo/mondo#disease_grouping>
    <http://purl.obolibrary.org/obo/mondo#matrix_grouping>
    <http://purl.obolibrary.org/obo/mondo#harrisons_view>
    <http://purl.obolibrary.org/obo/mondo#rare_grouping>
  }
  ?category_class oboInOwl:inSubset ?_subset .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_obsoletion_candidate(store: Store) -> Set[str]:
    """Get diseases that are obsoletion candidates."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#obsoletion_candidate> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_orphanet_subtype(store: Store) -> Set[str]:
    """Get diseases that correspond to Orphanet subtypes."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#ordo_subtype_of_a_disorder> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_orphanet_disorder(store: Store) -> Set[str]:
    """Get diseases that correspond to Orphanet disorders."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#ordo_disorder> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_icd_billable(store: Store) -> Set[str]:
    """Get diseases with billable ICD-10 codes."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#icd10_billable> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


# =============================================================================
# FILTER QUERIES - Hierarchy-based (rdfs:subClassOf*)
# =============================================================================


def _query_filter_paraphilic(store: Store) -> Set[str]:
    """Get paraphilic disorders (descendants of MONDO:0000596)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0000596 .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_cardiovascular(store: Store) -> Set[str]:
    """Get cardiovascular disorders (descendants of MONDO:0004995)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0004995 .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_heart_disorder(store: Store) -> Set[str]:
    """Get heart disorders (descendants of MONDO:0005267)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0005267 .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_inflammatory(store: Store) -> Set[str]:
    """Get inflammatory diseases (descendants of MONDO:0021166)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0021166 .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_psychiatric(store: Store) -> Set[str]:
    """Get psychiatric disorders (descendants of MONDO:0002025)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0002025 .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_cancer_or_benign_tumor(store: Store) -> Set[str]:
    """Get cancer or benign tumor (descendants of MONDO:0045024)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:subClassOf* MONDO:0045024 .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


# =============================================================================
# FILTER QUERIES - Label-based
# =============================================================================


def _query_filter_withorwithout(store: Store) -> Set[str]:
    """Get diseases with 'with or without' in the label."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:label ?label .
  FILTER(CONTAINS(?label, "with or without"))
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_andor(store: Store) -> Set[str]:
    """Get diseases with 'and/or' in the label."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:label ?label .
  FILTER(CONTAINS(?label, "and/or"))
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_acquired(store: Store) -> Set[str]:
    """Get diseases starting with 'acquired' in the label."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdfs:label ?label .
  FILTER(STRSTARTS(?label, "acquired "))
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


# =============================================================================
# FILTER QUERIES - Complex
# =============================================================================


def _query_filter_unclassified_hereditary(store: Store) -> Set[str]:
    """Get unclassified hereditary diseases (leaf nodes under hereditary disease with no other parents)."""
    query = (
        _get_sparql_prefixes()
        + """
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
    )
    return _query_and_extract_ids_from_column(store, query, column_name="entity")


def _query_filter_grouping_subset_ancestor(store: Store) -> Set[str]:
    """Get ancestors of designated grouping classes."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?grouping_class_subset rdfs:subClassOf+ ?category_class .
  VALUES ?_subset {
    <http://purl.obolibrary.org/obo/mondo#ordo_group_of_disorders>
    <http://purl.obolibrary.org/obo/mondo#disease_grouping>
    <http://purl.obolibrary.org/obo/mondo#harrisons_view>
    <http://purl.obolibrary.org/obo/mondo#rare_grouping>
  }
  ?grouping_class_subset oboInOwl:inSubset ?_subset .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_orphanet_subtype_descendant(store: Store) -> Set[str]:
    """Get descendants of Orphanet subtypes."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?category_class rdfs:subClassOf+ ?subtype_subset .
  ?subtype_subset oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#ordo_subtype_of_a_disorder> .
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_omimps(store: Store) -> Set[str]:
    """Get diseases corresponding to OMIM Phenotypic Series."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class skos:exactMatch ?match .
  FILTER(STRSTARTS(STR(?match), "https://omim.org/phenotypicSeries/PS"))
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_omimps_descendant(store: Store) -> Set[str]:
    """Get descendants of OMIM Phenotypic Series."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?category_class rdfs:subClassOf+ ?omimps .
  ?omimps skos:exactMatch ?match .
  FILTER(STRSTARTS(STR(?match), "https://omim.org/phenotypicSeries/PS"))
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_omim(store: Store) -> Set[str]:
    """Get diseases with OMIM entries."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class skos:exactMatch ?match .
  FILTER(STRSTARTS(STR(?match), "https://omim.org/entry/"))
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_leaf(store: Store) -> Set[str]:
    """Get leaf nodes (diseases with no children)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?category_class WHERE {
  ?category_class rdf:type owl:Class .
  FILTER NOT EXISTS {
    ?leaf_x rdfs:subClassOf ?category_class
  }
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_leaf_direct_parent(store: Store) -> Set[str]:
    """Get direct parents of leaf nodes."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?leaf rdfs:subClassOf ?category_class .
  FILTER NOT EXISTS {
    ?leaf_direct_parent_x rdfs:subClassOf ?leaf
  }
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


# =============================================================================
# FILTER QUERIES - ICD-10 based
# =============================================================================


def _query_filter_icd_category(store: Store) -> Set[str]:
    """Get diseases corresponding to ICD-10 categories (has . but not -)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?category_class oboInOwl:hasDbXref ?xref .
  ?a owl:annotatedSource ?category_class ;
     owl:annotatedProperty oboInOwl:hasDbXref ;
     owl:annotatedTarget ?xref ;
     oboInOwl:source ?type
  FILTER (
      STRSTARTS(str(?xref), "ICD10") &&
      !CONTAINS(SUBSTR(str(?xref), 6), "-") &&
      CONTAINS(SUBSTR(str(?xref), 6), ".")
  )
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_icd_chapter_code(store: Store) -> Set[str]:
    """Get diseases corresponding to ICD-10 chapter codes (has - but not .)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?category_class oboInOwl:hasDbXref ?xref .
  ?a owl:annotatedSource ?category_class ;
     owl:annotatedProperty oboInOwl:hasDbXref ;
     owl:annotatedTarget ?xref ;
     oboInOwl:source ?type
  FILTER (
      STRSTARTS(str(?xref), "ICD10") &&
      CONTAINS(SUBSTR(str(?xref), 6), "-") &&
      !CONTAINS(SUBSTR(str(?xref), 6), ".")
  )
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


def _query_filter_icd_chapter_header(store: Store) -> Set[str]:
    """Get diseases corresponding to ICD-10 chapter headers (no . or -)."""
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT ?category_class WHERE {
  ?category_class oboInOwl:hasDbXref ?xref .
  ?a owl:annotatedSource ?category_class ;
     owl:annotatedProperty oboInOwl:hasDbXref ;
     owl:annotatedTarget ?xref ;
     oboInOwl:source ?type
  FILTER (
      STRSTARTS(str(?xref), "ICD10") &&
      !CONTAINS(SUBSTR(str(?xref), 6), "-") &&
      !CONTAINS(SUBSTR(str(?xref), 6), ".")
  )
}
"""
    )
    return _query_and_extract_ids_from_column(store, query)


# =============================================================================
# Main Queries
# =============================================================================


def query_raw_disease_list_data_from_mondo(
    store: Store, billable_icd10: pd.DataFrame, df_subtypes: pd.DataFrame
) -> pd.DataFrame:
    """Assemble complete disease list with all metadata and filters.

    This is the main orchestrator function that runs all queries and assembles
    the results into a single DataFrame matching the schema of the original
    mega-query.

    Args:
        store: PyOxigraph Store object
        billable_icd10: DataFrame with billable ICD-10 codes to add to graph
        df_subtypes: DataFrame with subtype annotations to add to graph

    Returns:
        DataFrame with all disease list columns
    """
    logger.info("Assembling disease list from multiple focused queries")

    # 0. Postprocess Mondo
    _add_icd10_billable_data_to_graph(store, billable_icd10)
    _add_subtype_data_to_graph(store, df_subtypes)
    _postprocess_mondo_graph(store)

    # 1. Get base disease list
    df = _query_base_disease_list(store)
    logger.info(f"Base query returned {len(df)} diseases")

    # 2. Add metadata (left joins)
    logger.info("Adding metadata...")

    synonyms = _query_metadata_synonyms(store)
    df = df.merge(synonyms, on="category_class", how="left")

    subsets = _query_metadata_subsets(store)
    df = df.merge(subsets, on="category_class", how="left")

    crossrefs = _query_metadata_crossreferences(store)
    df = df.merge(crossrefs, on="category_class", how="left")

    malacards = _query_metadata_malacards_linkouts(store)
    df = df.merge(malacards, on="category_class", how="left")

    # 3. Add filter flags
    logger.info("Adding filter flags...")

    df["f_matrix_manually_included"] = (
        df["category_class"].isin(_query_filter_matrix_manually_included(store)).apply(lambda x: True if x else False)
    )

    df["f_matrix_manually_excluded"] = (
        df["category_class"].isin(_query_filter_matrix_manually_excluded(store)).apply(lambda x: True if x else False)
    )

    df["f_clingen"] = df["category_class"].isin(_query_filter_clingen(store)).apply(lambda x: True if x else False)

    df["f_susceptibility"] = (
        df["category_class"].isin(_query_filter_susceptibility(store)).apply(lambda x: True if x else False)
    )

    df["f_mondo_subtype"] = (
        df["category_class"].isin(_query_filter_mondo_subtype(store)).apply(lambda x: True if x else False)
    )

    df["f_pathway_defect"] = (
        df["category_class"].isin(_query_filter_pathway_defect(store)).apply(lambda x: True if x else False)
    )

    df["f_grouping_subset"] = (
        df["category_class"].isin(_query_filter_grouping_subset(store)).apply(lambda x: True if x else False)
    )

    df["f_obsoletion_candidate"] = (
        df["category_class"].isin(_query_filter_obsoletion_candidate(store)).apply(lambda x: True if x else False)
    )

    df["f_orphanet_subtype"] = (
        df["category_class"].isin(_query_filter_orphanet_subtype(store)).apply(lambda x: True if x else False)
    )

    df["f_orphanet_disorder"] = (
        df["category_class"].isin(_query_filter_orphanet_disorder(store)).apply(lambda x: True if x else False)
    )

    df["f_icd_billable"] = (
        df["category_class"].isin(_query_filter_icd_billable(store)).apply(lambda x: True if x else False)
    )

    df["f_paraphilic"] = df["category_class"].isin(_query_filter_paraphilic(store)).apply(lambda x: True if x else False)

    df["f_cardiovascular"] = (
        df["category_class"].isin(_query_filter_cardiovascular(store)).apply(lambda x: True if x else False)
    )

    df["f_filter_heart_disorder"] = (
        df["category_class"].isin(_query_filter_heart_disorder(store)).apply(lambda x: True if x else False)
    )

    df["f_inflammatory"] = (
        df["category_class"].isin(_query_filter_inflammatory(store)).apply(lambda x: True if x else False)
    )

    df["f_psychiatric"] = (
        df["category_class"].isin(_query_filter_psychiatric(store)).apply(lambda x: True if x else False)
    )

    df["f_cancer_or_benign_tumor"] = (
        df["category_class"].isin(_query_filter_cancer_or_benign_tumor(store)).apply(lambda x: True if x else False)
    )

    df["f_withorwithout"] = (
        df["category_class"].isin(_query_filter_withorwithout(store)).apply(lambda x: True if x else False)
    )

    df["f_andor"] = df["category_class"].isin(_query_filter_andor(store)).apply(lambda x: True if x else False)

    df["f_acquired"] = df["category_class"].isin(_query_filter_acquired(store)).apply(lambda x: True if x else False)

    df["f_unclassified_hereditary"] = (
        df["category_class"].isin(_query_filter_unclassified_hereditary(store)).apply(lambda x: True if x else False)
    )

    df["f_grouping_subset_ancestor"] = (
        df["category_class"].isin(_query_filter_grouping_subset_ancestor(store)).apply(lambda x: True if x else False)
    )

    df["f_orphanet_subtype_descendant"] = (
        df["category_class"].isin(_query_filter_orphanet_subtype_descendant(store)).apply(lambda x: True if x else False)
    )

    df["f_omimps"] = df["category_class"].isin(_query_filter_omimps(store)).apply(lambda x: True if x else False)

    df["f_omimps_descendant"] = (
        df["category_class"].isin(_query_filter_omimps_descendant(store)).apply(lambda x: True if x else False)
    )

    df["f_omim"] = df["category_class"].isin(_query_filter_omim(store)).apply(lambda x: True if x else False)

    df["f_leaf"] = df["category_class"].isin(_query_filter_leaf(store)).apply(lambda x: True if x else False)

    df["f_leaf_direct_parent"] = (
        df["category_class"].isin(_query_filter_leaf_direct_parent(store)).apply(lambda x: True if x else False)
    )

    df["f_icd_category"] = (
        df["category_class"].isin(_query_filter_icd_category(store)).apply(lambda x: True if x else False)
    )

    df["f_icd_chapter_code"] = (
        df["category_class"].isin(_query_filter_icd_chapter_code(store)).apply(lambda x: True if x else False)
    )

    df["f_icd_chapter_header"] = (
        df["category_class"].isin(_query_filter_icd_chapter_header(store)).apply(lambda x: True if x else False)
    )

    df = df.sort_values("label", ascending=False)

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

    if not child_id.startswith("MONDO:"):
        logging.error(f"Invalid child_id provided: {child_id}")
        return set()

    uri = child_id.replace("MONDO:", "http://purl.obolibrary.org/obo/MONDO_")

    query = (
        _get_sparql_prefixes()
        + f"""
        SELECT DISTINCT ?ancestor WHERE {{
            <{uri}> rdfs:subClassOf* ?ancestor .
            FILTER(?ancestor != <{uri}>)
        }}
    """
    )

    return _query_and_extract_ids_from_column(store, query, column_name="ancestor")


def query_get_descendants(store, root_id: str) -> set[str]:
    """Get all descendants (transitive children) of a node using SPARQL.

    Args:
        store: PyOxigraph Store object
        root_id: Root node ID (e.g., "MONDO:0000001")

    Returns:
        Set of descendant IDs in CURIE format
    """
    uri = root_id.replace("MONDO:", "http://purl.obolibrary.org/obo/MONDO_")

    # SPARQL query to find all descendants using property paths
    # rdfs:subClassOf* means zero or more subClassOf relationships
    query = (
        _get_sparql_prefixes()
        + f"""
        SELECT DISTINCT ?descendant WHERE {{
            ?descendant rdfs:subClassOf* <{uri}> .
            FILTER(?descendant != <{uri}>)
        }}
    """
    )

    return _query_and_extract_ids_from_column(store, query, column_name="descendant")


def query_mondo_labels(store: Store) -> pd.DataFrame:
    """Get MONDO labels.

    Args:
        store: PyOxigraph Store object

    Returns:
        DataFrame with columns: ID, LABEL
    """
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?ID ?LABEL

WHERE {
  ?ID rdfs:label ?LABEL .
  FILTER(STRSTARTS(STR(?ID), "http://purl.obolibrary.org/obo/MONDO_"))
}
"""
    )
    return _run_sparql_select(store, query)


def query_ontology_metadata(store: Store) -> pd.DataFrame:
    """Get ontology metadata.

    Args:
        store: PyOxigraph Store object

    Returns:
        DataFrame with columns: versionIRI, IRI, title
    """
    query = (
        _get_sparql_prefixes()
        + """
SELECT ?versionIRI ?IRI ?title

WHERE {
  ?IRI a owl:Ontology .
  OPTIONAL { ?IRI owl:versionIRI ?versionIRI . }
  OPTIONAL { ?IRI dc:title ?title . }
}
"""
    )
    return _run_sparql_select(store, query)


def query_mondo_obsoletes(store: Store) -> pd.DataFrame:
    """Get MONDO obsolete terms.

    Args:
        store: PyOxigraph Store object

    Returns:
        DataFrame with columns: cls, label, deprecated, replacements, considers
    """
    query = (
        _get_sparql_prefixes()
        + """
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

    OPTIONAL { ?cls IAO:0100001 ?repl. }
    OPTIONAL { ?cls oboInOwl:consider ?cons. }

    OPTIONAL { ?cls owl:deprecated ?deprecated. }
    FILTER(BOUND(?deprecated) &&
                 ( ?deprecated = true
                     || STR(?deprecated) = "true"
                     || ?deprecated = "true"^^xsd:boolean ))
}
GROUP BY ?cls ?label ?deprecated
ORDER BY ?cls
"""
    )
    return _run_sparql_select(store, query)


def query_matrix_disease_list_metrics(store: Store) -> pd.DataFrame:
    """Get disease metrics (descendant counts).

    Args:
        store: PyOxigraph Store object

    Returns:
        DataFrame with columns: category_class, count_descendants
    """
    query = (
        _get_sparql_prefixes()
        + """
SELECT DISTINCT
  ?category_class
  (COUNT(DISTINCT ?decendant_disease) AS ?count_descendants)
WHERE
{
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
    )
    return _run_sparql_select(store, query)


# =============================================================================
# Update Queries/Operations
# =============================================================================


def _add_icd10_billable_data_to_graph(mondo_graph: Store, billable_icd10: pd.DataFrame) -> None:
    """Add billable ICD-10 codes as RDF triples to the MONDO graph."""
    from pyoxigraph import DefaultGraph, Literal, NamedNode, Quad

    for _, row in billable_icd10.iterrows():
        subject = NamedNode(row["subject_id"].replace("MONDO:", "http://purl.obolibrary.org/obo/MONDO_"))

        subset_pred = NamedNode(row["predicate"])
        quad = Quad(subject, subset_pred, subset_pred, DefaultGraph())
        mondo_graph.add(quad)

        predicate = NamedNode("http://www.geneontology.org/formats/oboInOwl#hasDbXref")
        obj = Literal(row["object_id"])
        quad = Quad(subject, predicate, obj, DefaultGraph())
        mondo_graph.add(quad)

    logger.info(f"Added {len(billable_icd10)} billable ICD-10 dataframe rows to graph")


def _add_subtype_data_to_graph(mondo_graph: Store, df_subtypes: pd.DataFrame) -> None:
    """Add subtype annotations as RDF triples to the MONDO graph."""
    from pyoxigraph import DefaultGraph, NamedNode, Quad

    for _, row in df_subtypes.iterrows():
        subject = NamedNode(row["subject_id"].replace("MONDO:", "http://purl.obolibrary.org/obo/MONDO_"))

        # Add subset membership
        subset_pred = NamedNode(row["subset_predicate"])
        subset_obj = NamedNode(row["subset_object"])
        quad = Quad(subject, subset_pred, subset_obj, DefaultGraph())
        mondo_graph.add(quad)

        # Add contributor
        contrib_pred = NamedNode(row["contributor_predicate"])
        contrib_obj = NamedNode(row["contributor_object"])
        quad = Quad(subject, contrib_pred, contrib_obj, DefaultGraph())
        mondo_graph.add(quad)


def _postprocess_mondo_graph(mondo_graph: Store) -> None:
    """Apply SPARQL UPDATE transformations to the MONDO graph."""

    logger.info("Mondo postprocessing: Running SPARQL UPDATE transformations")
    update_queries = [
        ("inject-mondo-top-grouping", _query_inject_mondo_top_grouping()),
        ("inject-susceptibility-subset", _query_inject_susceptibility_subset()),
        ("inject-subset-declaration", _query_inject_subset_declaration()),
        ("downfill-disease-groupings", _query_downfill_disease_groupings()),
        ("disease-groupings-other", _query_disease_groupings_other()),
    ]

    for query_name, query_string in update_queries:
        logger.info(f"Running SPARQL UPDATE: {query_name}")
        mondo_graph.update(query_string)

    logger.info("Successfully applied all SPARQL transformations")


def _query_inject_mondo_top_grouping() -> str:
    """Get SPARQL UPDATE query to inject mondo_top_grouping subset."""
    return (
        _get_sparql_prefixes()
        + """
INSERT {
  ?subject oboInOwl:inSubset <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping> .
}
WHERE {
  ?subject rdfs:subClassOf <http://purl.obolibrary.org/obo/MONDO_0700096> .
}
"""
    )


def _query_inject_susceptibility_subset() -> str:
    """Get SPARQL UPDATE query to inject susceptibility subset annotations."""
    return (
        _get_sparql_prefixes()
        + """
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
    )


def _query_inject_subset_declaration() -> str:
    """Get SPARQL UPDATE query to declare all subsets as SubsetProperty."""
    return (
        _get_sparql_prefixes()
        + """
INSERT { ?y rdfs:subPropertyOf <http://www.geneontology.org/formats/oboInOwl#SubsetProperty> . }

WHERE {
  ?x <http://www.geneontology.org/formats/oboInOwl#inSubset>  ?y .
  FILTER(isIRI(?y))
  FILTER(regex(str(?y),"^(http://purl.obolibrary.org/obo/)") || regex(str(?y),"^(http://www.ebi.ac.uk/efo/)") || regex(str(?y),"^(https://w3id.org/biolink/)") || regex(str(?y),"^(http://purl.obolibrary.org/obo)"))
}
"""
    )


def _query_downfill_disease_groupings() -> str:
    """Get SPARQL UPDATE query to propagate groupings to descendants."""
    return (
        _get_sparql_prefixes()
        + """
INSERT {
  ?descendant oboInOwl:inSubset ?memberSubset .
  ?descendant oboInOwl:inSubset ?subjectSubset .
}
WHERE {
    VALUES ?subset {
      <http://purl.obolibrary.org/obo/mondo#mondo_txgnn>
      <http://purl.obolibrary.org/obo/mondo#harrisons_view>
      <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping>
      }
    ?subject oboInOwl:inSubset ?subset ; rdfs:label ?label .
    ?descendant rdfs:subClassOf+ ?subject .
    BIND(IRI(CONCAT(STR(?subset),"_member")) AS ?memberSubset)
    BIND(IRI(CONCAT(CONCAT(STR(?subset),"_"),REPLACE(LCASE(?label), "[^a-zA-Z0-9]", "_"))) AS ?subjectSubset)
}
"""
    )


def _query_disease_groupings_other() -> str:
    """Get SPARQL UPDATE query to mark ungrouped diseases as 'other'."""
    return (
        _get_sparql_prefixes()
        + """
INSERT {
  ?subject oboInOwl:inSubset ?otherSubset .
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

  BIND(IRI(CONCAT(STR(?subset), "_member")) AS ?memberSubset)
  BIND(IRI(CONCAT(STR(?subset), "_other")) AS ?otherSubset)

  FILTER NOT EXISTS {
    ?subject oboInOwl:inSubset ?subset .
  }
  FILTER NOT EXISTS {
    ?subject oboInOwl:inSubset ?memberSubset .
  }
}
"""
    )
