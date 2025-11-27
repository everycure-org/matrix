"""Disease list pipeline nodes for MATRIX.

This pipeline processes the MONDO disease ontology to create the MATRIX disease list,
which defines which diseases are included in the drug repurposing platform.

The workflow is modeled after the matrix-disease-list repository:
https://github.com/everycure-org/matrix-disease-list
"""

import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

import pandas as pd
import pandera.pandas as pa
from pyoxigraph import Store

logger = logging.getLogger(__name__)

@pa.check_output(
    pa.DataFrameSchema(
        {
            "mondo_id": pa.Column(dtype=str, nullable=False),
            "label": pa.Column(dtype=str, nullable=False),
            "comment": pa.Column(dtype=str, nullable=True),
            "issue": pa.Column(dtype=str, nullable=True),
            "obsoletion_date": pa.Column(dtype=str, nullable=False),
        },
        unique=["mondo_id"],
    )
)
def ingest_obsoletion_candidates(mondo_obsoletion_candidates: pd.DataFrame) -> pd.DataFrame:
    """Ingest MONDO obsoletion candidates DataFrame.

    Args:
        mondo_obsoletion_candidates: Raw DataFrame with obsoletion candidates

    Returns:
        Processed DataFrame with obsoletion candidates
    """
    return mondo_obsoletion_candidates

def _format_icd10_code_to_curie(code: str, prefix: str) -> str:
    """Format ICD-10 code as CURIE by inserting decimal point after third character."""
    if pd.notna(code) and len(code) > 3:
        return f"{prefix}:{code[:3]}.{code[3:]}"
    return code

@pa.check_input(
    pa.DataFrameSchema(
        {
            "CODE": pa.Column(dtype=str, nullable=False),
        },
        unique=["CODE"],
        strict=False,
    ), obj_getter="icd10_codes"
)
@pa.check_input(
    pa.DataFrameSchema(
        {
            "subject_id": pa.Column(dtype=str, nullable=False),
            "predicate_id": pa.Column(dtype=str, nullable=False),
            "object_id": pa.Column(dtype=str, nullable=False),
        },
        strict=False,
    ), obj_getter="mondo_sssom"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "subject_id": pa.Column(dtype=str, nullable=False),
            "predicate": pa.Column(dtype=str, nullable=False),
            "object_id": pa.Column(dtype=str, nullable=False),
        },
    )
)
def create_billable_icd10_codes(
    icd10_codes: pd.DataFrame,
    mondo_sssom: pd.DataFrame,
    icd10_billable_subset: str,
    icd10cm_prefix: str,
) -> pd.DataFrame:
    """Create dataframe for billable ICD-10-CM codes mapped to MONDO.

    Args:
        icd10_codes: DataFrame with ICD-10-CM codes (must have 'CODE' column)
        mondo_sssom: MONDO SSSOM mappings (must have 'predicate_id', 'object_id', 'subject_id')
        parameters: Dictionary containing various parameters from parameters.yml

    Returns:
        DataFrame with columns: subject_id (MONDO), predicate (subset URI), object_id (ICD10CM)
    """

    icd10_codes["code"] = icd10_codes["CODE"].apply(
        lambda code: _format_icd10_code_to_curie(code, icd10cm_prefix)
    )

    exact_matches = mondo_sssom[mondo_sssom["predicate_id"] == "skos:exactMatch"]

    icd10_data = exact_matches.merge(
        icd10_codes[["code"]], left_on="object_id", right_on="code", how="inner"
    )

    icd10_data["predicate"] = icd10_billable_subset
    return icd10_data[["subject_id", "predicate", "object_id"]]


def _is_ancestor(store: Store, parent_id: str, child_id: str) -> bool:
    """Check if parent_id is an ancestor of child_id in ontology hierarchy."""
    try:
        from matrix.pipelines.disease_list.queries import query_get_ancestors

        ancestors = query_get_ancestors(store, child_id)
        return parent_id in ancestors
    except Exception as e:
        logging.warning(f"Error checking relationship between {parent_id} and {child_id}: {e}")
        return False


def _compile_patterns(patterns_dict: Dict[str, str]) -> Dict[str, Pattern[str]]:
    """Compile regex patterns for better performance."""
    return {name: re.compile(pattern) for name, pattern in patterns_dict.items()}


def _find_parent_disease(
    label: str,
    disease_id: str,
    label_to_id_map: Dict[str, str],
    compiled_patterns: Dict[str, Pattern[str]],
    store: Store,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Match disease label against patterns to find parent disease with hierarchy validation."""
    for pattern_name, pattern in compiled_patterns.items():
        match = pattern.match(label)
        if not match:
            continue

        parent_label = match.group(1).strip().lower()
        parent_id = label_to_id_map.get(parent_label)

        if parent_id and _is_ancestor(store, parent_id, disease_id):
            return parent_label, parent_id, pattern_name

    return None, None, None


def _filter_mondo_labels(
    mondo_labels: pd.DataFrame, chromosomal_diseases: Set[str], human_diseases: Set[str], mondo_prefix: str
) -> pd.DataFrame:
    """Filter and prepare MONDO labels for subtype matching."""
    # The fact that chromosomal diseases are excluded has been requested by EveryCure
    # because they are likely to only _look_ like subtypes, but area actually full diseases
    # with references to chromosomes.
    filtered = (
        mondo_labels.dropna(subset=["LABEL"])
        .query(f"ID.str.startswith('{mondo_prefix}')")
        .loc[lambda df: ~df["ID"].isin(chromosomal_diseases)]
        .loc[lambda df: df["ID"].isin(human_diseases)][["ID", "LABEL"]]
        .rename(columns={"ID": "disease_id", "LABEL": "label"})
    )
    filtered["label_lower"] = filtered["label"].str.lower()
    return filtered


def _match_disease_subtypes(
    labels_df: pd.DataFrame, compiled_patterns: Dict[str, Pattern[str]], store: Store
) -> pd.DataFrame:
    """Apply pattern matching to all disease labels to identify subtypes and their parents."""
    label_to_id = dict(zip(labels_df["label_lower"], labels_df["disease_id"]))

    matches = labels_df.apply(
        lambda row: _find_parent_disease(row["label"], row["disease_id"], label_to_id, compiled_patterns, store),
        axis=1,
        result_type="expand",
    )
    matches.columns = ["parent_label", "parent_id", "pattern_name"]
    logger.info(f"Matched {matches['parent_id'].notna().sum()} diseases to parent groups ({len(matches)})")

    return pd.concat([labels_df, matches], axis=1)


def _build_subtype_counts(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Build counts of subtypes per parent disease."""
    valid_matches = matched_df.dropna(subset=["parent_label"])
    valid_matches = valid_matches[valid_matches["parent_label"].str.strip() != ""]
    logger.warning(f"Building subtype counts from {len(valid_matches)} valid matches")

    counts = valid_matches.groupby("parent_label").size().reset_index(name="count")

    result = valid_matches.merge(counts, on="parent_label")

    return result[["disease_id", "label", "parent_id", "parent_label", "count"]].rename(
        columns={
            "disease_id": "subset_id",
            "label": "subset_label",
            "parent_id": "subset_group_id",
            "parent_label": "subset_group_label",
            "count": "other_subsets_count",
        }
    )


def _build_subtype_df(matched_df: pd.DataFrame, mondo_subtype_subset: str, contributor: str) -> pd.DataFrame:
    """Convert matched subtype parents into annotation dataframe for graph enrichment."""
    unique_parents = matched_df[["parent_id", "parent_label"]].drop_duplicates().dropna().sort_values("parent_id")

    subset_rows = pd.DataFrame(
        {
            "subject_id": unique_parents["parent_id"],
            "subset_predicate": "http://www.geneontology.org/formats/oboInOwl#inSubset",
            "subset_object": mondo_subtype_subset,
            "contributor_predicate": "http://purl.org/dc/elements/1.1/contributor",
            "contributor_object": contributor,
        }
    )

    return subset_rows.reset_index(drop=True)


def _validate_mondo(mondo_graph, min_triples=1000000):
    q = "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"
    rows = list(mondo_graph.query(q))
    if not rows:
        raise ValueError("MONDO graph query returned no results")
    
    triple_count = int(rows[0]["count"].value) if rows else 0
    
    if triple_count < min_triples:
        raise ValueError(
            f"MONDO graph seems too small: only {triple_count} triples "
            f"(expected ≥ {min_triples})."
        )
    logger.info(f"MONDO graph validated with {triple_count} triples")
    return triple_count


@pa.check_input(
    pa.DataFrameSchema(
        {
            "subject_id": pa.Column(dtype=str, nullable=False),
            "predicate": pa.Column(dtype=str, nullable=False),
            "object_id": pa.Column(dtype=str, nullable=False),
        },
        strict=True,
    ), obj_getter="billable_icd10"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "versionIRI": pa.Column(dtype=str, nullable=False),
            "IRI": pa.Column(dtype=str, nullable=False),
            "title": pa.Column(dtype=str, nullable=False),
        },
        strict=True,
    ), obj_getter="mondo_metadata"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "cls": pa.Column(dtype=str, nullable=False),
            "label": pa.Column(dtype=str, nullable=False),
            "deprecated": pa.Column(dtype=bool, nullable=False),
            "replacements": pa.Column(dtype=str, nullable=False),
            "considers": pa.Column(dtype=str, nullable=False),
        },
        strict=True,
    ), obj_getter="mondo_obsoletes"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "category_class": pa.Column(dtype=str, nullable=False),
            "label": pa.Column(dtype=str, nullable=False),
            "definition": pa.Column(dtype=str, nullable=True),
            "synonyms": pa.Column(dtype=str, nullable=True),
            "subsets": pa.Column(dtype=str, nullable=True),
            "crossreferences": pa.Column(dtype=str, nullable=True),
            "malacards_linkouts": pa.Column(dtype=str, nullable=True),
            "f_matrix_manually_included": pa.Column(dtype=bool, nullable=False),
            "f_matrix_manually_excluded": pa.Column(dtype=bool, nullable=False),
            "f_clingen": pa.Column(dtype=bool, nullable=False),
            "f_susceptibility": pa.Column(dtype=bool, nullable=False),
            "f_mondo_subtype": pa.Column(dtype=bool, nullable=False),
            "f_pathway_defect": pa.Column(dtype=bool, nullable=False),
            "f_grouping_subset": pa.Column(dtype=bool, nullable=False),
            "f_obsoletion_candidate": pa.Column(dtype=bool, nullable=False),
            "f_orphanet_subtype": pa.Column(dtype=bool, nullable=False),
            "f_orphanet_disorder": pa.Column(dtype=bool, nullable=False),
            "f_icd_billable": pa.Column(dtype=bool, nullable=False),
            "f_paraphilic": pa.Column(dtype=bool, nullable=False),
            "f_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "f_filter_heart_disorder": pa.Column(dtype=bool, nullable=False),
            "f_inflammatory": pa.Column(dtype=bool, nullable=False),
            "f_psychiatric": pa.Column(dtype=bool, nullable=False),
            "f_cancer_or_benign_tumor": pa.Column(dtype=bool, nullable=False),
            "f_withorwithout": pa.Column(dtype=bool, nullable=False),
            "f_andor": pa.Column(dtype=bool, nullable=False),
            "f_acquired": pa.Column(dtype=bool, nullable=False),
            "f_unclassified_hereditary": pa.Column(dtype=bool, nullable=False),
            "f_grouping_subset_ancestor": pa.Column(dtype=bool, nullable=False),
            "f_orphanet_subtype_descendant": pa.Column(dtype=bool, nullable=False),
            "f_omimps": pa.Column(dtype=bool, nullable=False),
            "f_omimps_descendant": pa.Column(dtype=bool, nullable=False),
            "f_omim": pa.Column(dtype=bool, nullable=False),
            "f_leaf": pa.Column(dtype=bool, nullable=False),
            "f_leaf_direct_parent": pa.Column(dtype=bool, nullable=False),
            "f_icd_category": pa.Column(dtype=bool, nullable=False),
            "f_icd_chapter_code": pa.Column(dtype=bool, nullable=False),
            "f_icd_chapter_header": pa.Column(dtype=bool, nullable=False),
            
        },
        strict=True,
    ), obj_getter="disease_list_raw"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "category_class": pa.Column(dtype=str, nullable=False),
            "count_descendants": pa.Column(dtype=int, nullable=False),  
        },
        strict=True,
    ), obj_getter="mondo_metrics"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "subset_id": pa.Column(dtype=str, nullable=False),
            "subset_label": pa.Column(dtype=str, nullable=False),
            "subset_group_id": pa.Column(dtype=str, nullable=False),
            "subset_group_label": pa.Column(dtype=str, nullable=False),
            "other_subsets_count": pa.Column(dtype=int, nullable=False),
        },
        checks=[
            pa.Check(
                lambda df: df.shape[0] >= 100,
                name="min_rows",
                error=f"subtype_counts must have at least 100 rows",
            )
        ],
        strict=True,
    ), obj_getter="subtype_counts"
)
def extract_disease_data_from_mondo(
    mondo_graph: Store,
    billable_icd10: pd.DataFrame,
    subtypes_params: Dict[str, Any],
    subtype_patterns: Dict[str, str],
) -> Dict[str, Any]:
    """Extract all disease-related data from MONDO ontology.

    This function performs the complete MONDO processing pipeline:
    1. Extract metadata (version, labels, obsoletes)
    2. Identify disease subtypes using hierarchy analysis
    3. Enrich MONDO with subtype annotations and billable ICD-10 codes
    4. Extract disease list data and metrics

    All intermediate data (labels, dataframes) are kept in-memory for efficiency.

    Args:
        mondo_graph: PyOxigraph Store object containing MONDO ontology
        billable_icd10: Billable ICD-10 dataframe
        subtypes_params: Dictionary containing subtype identification parameters
        subtype_patterns: Dictionary of regex patterns for matching disease subtypes

    Returns:
        Dictionary containing:
            - mondo_metadata: DataFrame with MONDO version info
            - mondo_obsoletes: DataFrame with obsolete terms
            - disease_list_raw: DataFrame with raw disease list features
            - mondo_metrics: DataFrame with disease metrics
            - mondo_preprocessed: PyOxigraph Store with preprocessed ontology
            - subtype_counts: DataFrame with subtype counts (for downstream filtering)
    """
    logger.info("Extracting disease data from MONDO ontology")

    from matrix.pipelines.disease_list.queries import (
        query_matrix_disease_list_metrics, query_mondo_labels,
        query_mondo_obsoletes, query_ontology_metadata,
        query_raw_disease_list_data_from_mondo)

    _validate_mondo(mondo_graph)

    logger.info("Step 0: Extracting metadata and labels from MONDO")

    mondo_labels = query_mondo_labels(mondo_graph)
    logger.info(f"Extracted {len(mondo_labels)} MONDO labels")

    mondo_metadata = query_ontology_metadata(mondo_graph)
    logger.info(f"Extracted MONDO metadata: {mondo_metadata}")

    mondo_obsoletes = query_mondo_obsoletes(mondo_graph)
    mondo_obsoletes["deprecated"] = mondo_obsoletes["deprecated"].str.lower() == "true"
    logger.info(f"Extracted {len(mondo_obsoletes)} obsolete MONDO terms")

    logger.info("Step 1: Identifying disease subtypes")

    subtype_counts, df_subtypes = _extract_subtype_data(mondo_graph, subtypes_params, subtype_patterns, mondo_labels)

    logger.info("Step 2: Enriching MONDO graph with annotations")

    _validate_mondo(mondo_graph)

    logger.info("Step 3: Extracting disease data from enriched MONDO")
    disease_list_raw = query_raw_disease_list_data_from_mondo(mondo_graph, billable_icd10, df_subtypes)
    logger.info(f"Extracted {len(disease_list_raw)} diseases in raw list")

    logger.info("Extracting disease metrics")
    mondo_metrics = query_matrix_disease_list_metrics(mondo_graph)
    mondo_metrics["count_descendants"] = (
        pd.to_numeric(mondo_metrics["count_descendants"], errors="raise")
        .astype("int64")
    )
    logger.info(f"Extracted metrics for {len(mondo_metrics)} diseases")

    _validate_mondo(mondo_graph)

    logger.info("Finished extracting all disease data from MONDO")
    return {
        "mondo_metadata": mondo_metadata,
        "mondo_obsoletes": mondo_obsoletes,
        "disease_list_raw": disease_list_raw,
        "mondo_metrics": mondo_metrics,
        "subtype_counts": subtype_counts,
    }


def _extract_subtype_data(
    mondo_graph: Store, subtypes_params: Dict[str, Any], subtype_patterns: Dict[str, str], mondo_labels: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and identify disease subtypes using pattern matching and hierarchy validation."""
    from matrix.pipelines.disease_list.queries import query_get_descendants

    chromosomal_diseases_root = subtypes_params["chromosomal_diseases_root"]
    chromosomal_diseases_exceptions = subtypes_params["chromosomal_diseases_exceptions"]
    logger.info(f"Getting descendants of {chromosomal_diseases_root} using SPARQL")
    chromosomal_diseases = query_get_descendants(mondo_graph, chromosomal_diseases_root)
    chromosomal_diseases.add(chromosomal_diseases_root)
    for exception_id in chromosomal_diseases_exceptions:
        chromosomal_diseases.discard(exception_id)
    logger.info(f"Extracted {len(chromosomal_diseases)} chromosomal diseases")

    human_diseases_root = subtypes_params["human_diseases_root"]
    logger.info(f"Getting descendants of {human_diseases_root} using SPARQL")
    human_diseases = query_get_descendants(mondo_graph, human_diseases_root)
    human_diseases.add(human_diseases_root)

    logger.info(f"Found {len(chromosomal_diseases)} chromosomal diseases and {len(human_diseases)} human diseases")

    mondo_prefix = subtypes_params["mondo_prefix"]
    filtered_labels = _filter_mondo_labels(mondo_labels, chromosomal_diseases, human_diseases, mondo_prefix)
    compiled_patterns = _compile_patterns(subtype_patterns)
    matched = _match_disease_subtypes(filtered_labels, compiled_patterns, mondo_graph)
    subtype_counts = _build_subtype_counts(matched)
    mondo_subtype_subset = subtypes_params["mondo_subtype_subset"]
    default_contributor = subtypes_params["default_contributor"]

    df_subtypes = _build_subtype_df(matched, mondo_subtype_subset, default_contributor)
    logger.info(f"Identified {len(subtype_counts)} disease subtypes")

    return subtype_counts, df_subtypes


# TODO the following function might not be needed anymore. It basically sets the "official_matrix_filter" which is not used in production
def _matrix_disease_filter(df_disease_list_unfiltered: pd.DataFrame) -> pd.DataFrame:
    """Apply MATRIX filter rules to determine which diseases are included in the platform.

    Args:
        df_disease_list_unfiltered: Disease list with filter feature columns

    Returns:
        DataFrame with 'official_matrix_filter' column added
    """
    df_disease_list_unfiltered = df_disease_list_unfiltered.reset_index(drop=True)

    filter_column = "official_matrix_filter"

    df_disease_list_unfiltered[filter_column] = False

    conflicts = df_disease_list_unfiltered[
        (df_disease_list_unfiltered["f_matrix_manually_included"])
        & (df_disease_list_unfiltered["f_matrix_manually_excluded"])
    ]

    if not conflicts.empty:
        conflict_str = conflicts.to_string(index=False)
        raise ValueError(
            f"Conflicts found: The following entries are marked as both manually included and manually excluded:\n{conflict_str}"
        )

    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_matrix_manually_included"]
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_leaf"]

    df_disease_list_unfiltered[filter_column] |= (df_disease_list_unfiltered["f_leaf_direct_parent"]) & (
        (df_disease_list_unfiltered["f_omim"])
        | (df_disease_list_unfiltered["f_omimps_descendant"])
        | (df_disease_list_unfiltered["f_icd_category"])
        | (df_disease_list_unfiltered["f_orphanet_disorder"])
        | (df_disease_list_unfiltered["f_orphanet_subtype"])
    )

    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_icd_category"]
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_orphanet_disorder"]
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_clingen"]
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_omim"]

    df_disease_list_unfiltered.loc[df_disease_list_unfiltered["f_unclassified_hereditary"], filter_column] = (
        False
    )
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered["f_paraphilic"], filter_column] = False
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered["f_matrix_manually_excluded"], filter_column] = (
        False
    )

    return df_disease_list_unfiltered


def _extract_groupings(
    subsets: str, groupings: List[str], subset_prefix: str, subset_delimiter: str, grouping_delimiter: str
) -> Dict[str, str]:
    """Extract groupings from semicolon-delimited subset string.

    Args:
        subsets: Semicolon-delimited string of subsets
        groupings: List of grouping names to extract
        subset_prefix: Prefix for subset names (e.g., "mondo:")
        subset_delimiter: Delimiter between subsets (e.g., ";")
        grouping_delimiter: Delimiter for joining multiple values (e.g., "|")

    Returns:
        Dictionary mapping grouping names to extracted values
    """
    result: Dict[str, List[str]] = {grouping: [] for grouping in groupings}

    if subsets:
        for subset in subsets.split(subset_delimiter):
            subset = subset.strip()
            for grouping in groupings:
                if subset.startswith(f"{subset_prefix}{grouping}"):
                    subset_tag = (
                        subset.replace(subset_prefix, "")
                        .replace(grouping, "")
                        .replace(" ", "")
                        .strip("_")
                        .replace(grouping_delimiter, "")
                    )
                    if subset_tag and subset_tag != "member":
                        result[grouping].append(subset_tag)

    return {
        key: grouping_delimiter.join([v for v in values if v != "other"] if len(values) > 1 else values)
        if values
        else ""
        for key, values in result.items()
    }


def _is_grouping_heuristic(
    df: pd.DataFrame, grouping_columns: List[str], not_grouping_columns: List[str], output_column: str
) -> pd.DataFrame:
    """Apply grouping heuristic to classify diseases as groupings or specific diseases.

    The heuristic works by checking positive indicators (grouping_columns) and negative
    indicators (not_grouping_columns). A disease is marked as a grouping if any grouping
    column is True, UNLESS any not_grouping column is also True (which takes precedence).

    Args:
        df: Input DataFrame with disease data
        grouping_columns: List of columns that indicate a disease is a grouping
        not_grouping_columns: List of columns that override grouping classification
        output_column: Name of output column for heuristic result

    Returns:
        DataFrame with new heuristic column added
    """
    df = df.copy()

    expected_cols = set(grouping_columns) | set(not_grouping_columns)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing columns: {df.columns}")
        raise ValueError(f"DataFrame is missing expected grouping columns: {', '.join(sorted(missing))}")

    df[output_column] = False

    for col in grouping_columns:
        df[output_column] = df[output_column] | df[col].fillna(False)

    for col in not_grouping_columns:
        if col not in df.columns:
            logging.warning(f"Missing not-grouping column: {col}")
            continue
        df.loc[df[col].fillna(False), output_column] = False

    return df


def _prepare_subtype_counts(subtype_counts: pd.DataFrame) -> pd.DataFrame:
    """Prepare subtype counts DataFrame for merging.

    Args:
        subtype_counts: DataFrame with subtype information

    Returns:
        Simplified DataFrame with category_class and count_subtypes columns
    """
    filtered_counts = subtype_counts[
        subtype_counts["subset_group_id"].notna() & (subtype_counts["subset_group_id"] != "")
    ]

    return (
        filtered_counts[["subset_group_id", "other_subsets_count"]]
        .drop_duplicates()
        .rename(columns={"subset_group_id": "category_class", "other_subsets_count": "count_subtypes"})
    )


def _extract_and_pivot_groupings(
    df: pd.DataFrame, groupings: List[str], subset_prefix: str, subset_delimiter: str, grouping_delimiter: str
) -> pd.DataFrame:
    """Extract and pivot disease groupings from subset annotations.

    Args:
        df: DataFrame with category_class, label, and subsets columns
        groupings: List of grouping names to extract
        subset_prefix: Prefix for subset names
        subset_delimiter: Delimiter between subsets
        grouping_delimiter: Delimiter for joining multiple values

    Returns:
        DataFrame with category_class and one column per grouping
    """
    groupings_extracted = (
        df["subsets"]
        .apply(lambda x: _extract_groupings(x, groupings, subset_prefix, subset_delimiter, grouping_delimiter))
        .apply(pd.Series)
    )

    result = pd.concat([df[["category_class"]], groupings_extracted], axis=1)
    return result.sort_values("category_class")


def _merge_disease_data_sources(
    base_df: pd.DataFrame,
    groupings_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    subtype_counts_df: pd.DataFrame,
    full_subtype_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all disease data sources into single DataFrame.

    This optimizes 4 separate merges into a more efficient operation.

    Args:
        base_df: Base disease list with filters applied
        groupings_df: Disease groupings pivoted data
        metrics_df: MONDO metrics data
        subtype_counts_df: Simplified subtype counts
        full_subtype_counts: Full subtype information

    Returns:
        Merged DataFrame with all data sources
    """
    merged = base_df.merge(groupings_df, on="category_class", how="left")

    metrics_df_filtered = metrics_df[
        metrics_df["category_class"].notna() & (metrics_df["category_class"] != "")
    ]
    subtype_counts_df_filtered = subtype_counts_df[
        subtype_counts_df["category_class"].notna() & (subtype_counts_df["category_class"] != "")
    ]

    metrics_and_counts = metrics_df_filtered.merge(subtype_counts_df_filtered, on="category_class", how="outer")
    merged = merged.merge(metrics_and_counts, on="category_class", how="left")

    full_subtype_counts_filtered = full_subtype_counts[
        full_subtype_counts["subset_id"].notna() & (full_subtype_counts["subset_id"] != "")
    ]

    merged = merged.merge(
        full_subtype_counts_filtered[["subset_id", "subset_group_id", "subset_group_label", "other_subsets_count"]],
        left_on="category_class",
        right_on="subset_id",
        how="left",
    )

    return merged.drop(columns=["subset_id"], errors="ignore")


@pa.check_input(
    pa.DataFrameSchema(
        {
            "category_class": pa.Column(dtype=str, nullable=False),
            "label": pa.Column(dtype=str, nullable=False),
            "definition": pa.Column(dtype=str, nullable=True),
            "synonyms": pa.Column(dtype=str, nullable=True),
            "subsets": pa.Column(dtype=str, nullable=True),
            "crossreferences": pa.Column(dtype=str, nullable=True),
            "malacards_linkouts": pa.Column(dtype=str, nullable=True),
            "f_matrix_manually_included": pa.Column(dtype=bool, nullable=False),
            "f_matrix_manually_excluded": pa.Column(dtype=bool, nullable=False),
            "f_clingen": pa.Column(dtype=bool, nullable=False),
            "f_susceptibility": pa.Column(dtype=bool, nullable=False),
            "f_mondo_subtype": pa.Column(dtype=bool, nullable=False),
            "f_pathway_defect": pa.Column(dtype=bool, nullable=False),
            "f_grouping_subset": pa.Column(dtype=bool, nullable=False),
            "f_obsoletion_candidate": pa.Column(dtype=bool, nullable=False),
            "f_orphanet_subtype": pa.Column(dtype=bool, nullable=False),
            "f_orphanet_disorder": pa.Column(dtype=bool, nullable=False),
            "f_icd_billable": pa.Column(dtype=bool, nullable=False),
            "f_paraphilic": pa.Column(dtype=bool, nullable=False),
            "f_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "f_filter_heart_disorder": pa.Column(dtype=bool, nullable=False),
            "f_inflammatory": pa.Column(dtype=bool, nullable=False),
            "f_psychiatric": pa.Column(dtype=bool, nullable=False),
            "f_cancer_or_benign_tumor": pa.Column(dtype=bool, nullable=False),
            "f_withorwithout": pa.Column(dtype=bool, nullable=False),
            "f_andor": pa.Column(dtype=bool, nullable=False),
            "f_acquired": pa.Column(dtype=bool, nullable=False),
            "f_unclassified_hereditary": pa.Column(dtype=bool, nullable=False),
            "f_grouping_subset_ancestor": pa.Column(dtype=bool, nullable=False),
            "f_orphanet_subtype_descendant": pa.Column(dtype=bool, nullable=False),
            "f_omimps": pa.Column(dtype=bool, nullable=False),
            "f_omimps_descendant": pa.Column(dtype=bool, nullable=False),
            "f_omim": pa.Column(dtype=bool, nullable=False),
            "f_leaf": pa.Column(dtype=bool, nullable=False),
            "f_leaf_direct_parent": pa.Column(dtype=bool, nullable=False),
            "f_icd_category": pa.Column(dtype=bool, nullable=False),
            "f_icd_chapter_code": pa.Column(dtype=bool, nullable=False),
            "f_icd_chapter_header": pa.Column(dtype=bool, nullable=False),
            
        },
        strict=True,
    ), obj_getter="disease_list_raw"
)
@pa.check_input(
    pa.DataFrameSchema(
        {
            "category_class": pa.Column(dtype=str, nullable=False),
            "count_descendants": pa.Column(dtype=int, nullable=False),  
        },
        strict=True,
    ), obj_getter="mondo_metrics"
)
@pa.check_input(
    pa.DataFrameSchema(
        {
            "subset_id": pa.Column(dtype=str, nullable=False),
            "subset_label": pa.Column(dtype=str, nullable=False),
            "subset_group_id": pa.Column(dtype=str, nullable=False),
            "subset_group_label": pa.Column(dtype=str, nullable=False),
            "other_subsets_count": pa.Column(dtype=int, nullable=False),
        },
        strict=True,
    ), obj_getter="subtype_counts"
)
@pa.check_output(
    pa.DataFrameSchema(
        {
            "category_class": pa.Column(dtype=str, nullable=False),
            "label": pa.Column(dtype=str, nullable=False),
            "definition": pa.Column(dtype=str, nullable=True),
            "synonyms": pa.Column(dtype=str, nullable=True),
            "subsets": pa.Column(dtype=str, nullable=True),
            "crossreferences": pa.Column(dtype=str, nullable=True),
            "malacards_linkouts": pa.Column(dtype=str, nullable=True),
            "is_matrix_manually_included": pa.Column(dtype=bool, nullable=False),
            "is_matrix_manually_excluded": pa.Column(dtype=bool, nullable=False),
            "is_clingen": pa.Column(dtype=bool, nullable=False),
            "is_susceptibility": pa.Column(dtype=bool, nullable=False),
            "is_mondo_subtype": pa.Column(dtype=bool, nullable=False),
            "is_pathway_defect": pa.Column(dtype=bool, nullable=False),
            "is_grouping_subset": pa.Column(dtype=bool, nullable=False),
            "is_obsoletion_candidate": pa.Column(dtype=bool, nullable=False),
            "is_orphanet_subtype": pa.Column(dtype=bool, nullable=False),
            "is_orphanet_disorder": pa.Column(dtype=bool, nullable=False),
            "is_icd_billable": pa.Column(dtype=bool, nullable=False),
            "is_paraphilic": pa.Column(dtype=bool, nullable=False),
            "is_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "is_filter_heart_disorder": pa.Column(dtype=bool, nullable=False),
            "is_inflammatory": pa.Column(dtype=bool, nullable=False),
            "is_psychiatric": pa.Column(dtype=bool, nullable=False),
            "is_cancer_or_benign_tumor": pa.Column(dtype=bool, nullable=False),
            "is_withorwithout": pa.Column(dtype=bool, nullable=False),
            "is_andor": pa.Column(dtype=bool, nullable=False),
            "is_acquired": pa.Column(dtype=bool, nullable=False),
            "is_unclassified_hereditary": pa.Column(dtype=bool, nullable=False),
            "is_grouping_subset_ancestor": pa.Column(dtype=bool, nullable=False),
            "is_orphanet_subtype_descendant": pa.Column(dtype=bool, nullable=False),
            "is_omimps": pa.Column(dtype=bool, nullable=False),
            "is_omimps_descendant": pa.Column(dtype=bool, nullable=False),
            "is_omim": pa.Column(dtype=bool, nullable=False),
            "is_leaf": pa.Column(dtype=bool, nullable=False),
            "is_leaf_direct_parent": pa.Column(dtype=bool, nullable=False),
            "is_icd_category": pa.Column(dtype=bool, nullable=False),
            "is_icd_chapter_code": pa.Column(dtype=bool, nullable=False),
            "is_icd_chapter_header": pa.Column(dtype=bool, nullable=False),
            "official_matrix_filter": pa.Column(dtype=bool, nullable=False),
            "harrisons_view": pa.Column(dtype=str, nullable=False),
            "mondo_txgnn": pa.Column(dtype=str, nullable=False),
            "mondo_top_grouping": pa.Column(dtype=str, nullable=False),
            "medical_specialization": pa.Column(dtype=str, nullable=False),
            "txgnn": pa.Column(dtype=str, nullable=False),
            "anatomical": pa.Column(dtype=str, nullable=False),
            "is_pathogen_caused": pa.Column(dtype=str, nullable=False),
            "is_cancer": pa.Column(dtype=str, nullable=False),
            "is_glucose_dysfunction": pa.Column(dtype=str, nullable=False),
            "tag_existing_treatment": pa.Column(dtype=str, nullable=False),
            "tag_qaly_lost": pa.Column(dtype=str, nullable=False),
            "count_descendants": pa.Column(dtype=int, nullable=False),
            "count_subtypes": pa.Column(dtype=int, nullable=False),
            "subset_group_id": pa.Column(dtype=str, nullable=True),
            "subset_group_label": pa.Column(dtype=str, nullable=True),
            "other_subsets_count": pa.Column(dtype=int, nullable=True),
            "count_descendants_without_subtypes": pa.Column(dtype=int, nullable=False),
            "is_grouping_heuristic": pa.Column(dtype=bool, nullable=False),
        },
        strict=True,
        unique=["category_class"],
        checks=[
            pa.Check(lambda df: df.shape[0] >= 15000, name="min_rows", error="disease_list must have at least 15000 rows")
            ],
    ), obj_getter="disease_list"
)

def create_disease_list(
    disease_list_raw: pd.DataFrame,
    mondo_metrics: pd.DataFrame,
    subtype_counts: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Apply filters and transformations to create final disease list.

    Args:
        disease_list_raw: Raw disease list with filter features
        mondo_metrics: MONDO metrics DataFrame
        subtype_counts: Subtype counts and relationships
        parameters: Dictionary containing:
            - curated_groupings: List of curated disease grouping names
            - llm_groupings: List of LLM-generated grouping names
            - subset_prefix: Prefix for subset parsing (e.g., "mondo:")
            - subset_delimiter: Delimiter for subset strings (e.g., ";")
            - grouping_delimiter: Delimiter for joining grouping values (e.g., "|")
            - grouping_columns: Columns indicating a disease is a grouping
            - not_grouping_columns: Columns that override grouping classification
            - grouping_heuristic_column: Name of output heuristic column

    Returns:
        Dictionary with 'disease_list' key containing final DataFrame
    """
    logger.info("Create final disease list")

    curated_groupings = parameters["curated_groupings"]
    llm_groupings = parameters["llm_groupings"]
    subset_prefix = "mondo:"
    subset_delimiter = ";"
    grouping_delimiter = "|"
    grouping_columns = parameters["grouping_columns"]
    not_grouping_columns = parameters["not_grouping_columns"]
    grouping_heuristic_column = parameters["grouping_heuristic_column"]

    subtype_group_counts = _prepare_subtype_counts(subtype_counts)
    filtered_df = _matrix_disease_filter(disease_list_raw)
    # See: https://github.com/everycure-org/matrix-disease-list/issues/75
    filtered_df = filtered_df.rename(columns=lambda x: re.sub(r"^f_", "is_", x) if x.startswith("f_") else x)

    all_groupings = curated_groupings + llm_groupings
    groupings_df = _extract_and_pivot_groupings(
        filtered_df[["category_class", "label", "subsets"]],
        all_groupings,
        subset_prefix,
        subset_delimiter,
        grouping_delimiter,
    )

    merged_df = _merge_disease_data_sources(
        filtered_df, groupings_df, mondo_metrics, subtype_group_counts, subtype_counts
    )

    merged_df["count_subtypes"] = (
        pd.to_numeric(merged_df["count_subtypes"], errors="coerce").fillna(0).astype(int)
    )
    merged_df["count_descendants"] = (
        pd.to_numeric(merged_df["count_descendants"], errors="coerce").fillna(0).astype(int)
    )
    merged_df["other_subsets_count"] = (
        pd.to_numeric(merged_df["other_subsets_count"], errors="coerce").astype("Int64")
    )
    merged_df["count_descendants_without_subtypes"] = merged_df["count_descendants"] - merged_df["count_subtypes"]

    merged_df = _is_grouping_heuristic(merged_df, grouping_columns, not_grouping_columns, grouping_heuristic_column)

    logger.info(f"Created disease list with {len(merged_df)} entries")

    return {"disease_list": merged_df}
