"""Disease list pipeline nodes for MATRIX.

This pipeline processes the MONDO disease ontology to create the MATRIX disease list,
which defines which diseases are included in the drug repurposing platform.

The workflow is modeled after the matrix-disease-list repository:
https://github.com/everycure-org/matrix-disease-list
"""

import logging
import re
from typing import Any, Dict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _format_icd10_code_to_curie(code: str, prefix: str) -> str:
    """Convert ICD-10 code to CURIE format.

    Args:
        code: ICD-10 code (e.g., "A001")
        prefix: CURIE prefix (e.g., "ICD10CM")

    Returns:
        CURIE formatted code (e.g., "ICD10CM:A00.1")
    """
    if pd.notna(code) and len(code) > 3:
        return f"{prefix}:{code[:3]}.{code[3:]}"
    return code


def create_billable_icd10_codes(
    icd10_codes: pd.DataFrame,
    mondo_sssom: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Create dataframe for billable ICD-10-CM codes mapped to MONDO.

    Args:
        icd10_codes: DataFrame with ICD-10-CM codes (must have 'CODE' column)
        mondo_sssom: MONDO SSSOM mappings (must have 'predicate_id', 'object_id', 'subject_id')
        parameters: Dictionary containing various parameters from parameters.yml

    Returns:
        DataFrame with columns: subject_id (MONDO), predicate (subset URI), object_id (ICD10CM)
    """
    logger.info("Creating billable ICD-10-CM dataframe")

    # Extract parameters
    exact_match_predicate = parameters["exact_match_predicate"]
    icd10_billable_subset = parameters["icd10_billable_subset"]
    icd10cm_prefix = parameters["icd10cm_prefix"]

    # Convert ICD-10 codes to CURIE format without mutating input
    icd10_codes_formatted = icd10_codes.copy()
    icd10_codes_formatted["CODE"] = icd10_codes_formatted["CODE"].apply(
        lambda code: _format_icd10_code_to_curie(code, icd10cm_prefix)
    )

    # Filter SSSOM mappings to only exact matches
    exact_matches = mondo_sssom[mondo_sssom["predicate_id"] == exact_match_predicate].copy()

    # Find billable codes that have MONDO mappings
    billable_with_mappings = icd10_codes_formatted[icd10_codes_formatted["CODE"].isin(exact_matches["object_id"])]

    logger.info(f"Found {len(billable_with_mappings)} billable ICD-10 codes with MONDO mappings")

    # Create dataframe rows by joining billable codes with their MONDO mappings
    icd10_data = exact_matches.merge(
        billable_with_mappings[["CODE"]], left_on="object_id", right_on="CODE", how="inner"
    )

    # Return simple DataFrame (no ROBOT header)
    return pd.DataFrame(
        {
            "subject_id": icd10_data["subject_id"],
            "predicate": icd10_billable_subset,
            "object_id": icd10_data["object_id"],
        }
    )


def _is_ancestor(store, parent_id, child_id):
    """Check if parent_id is an ancestor of child_id in the ontology.

    Args:
        store: PyOxigraph Store object
        parent_id: Parent node ID (e.g., "MONDO:0000001")
        child_id: Child node ID (e.g., "MONDO:0000002")

    Returns:
        True if parent_id is an ancestor of child_id, False otherwise
    """
    try:
        from matrix.pipelines.disease_list.queries import query_get_ancestors

        ancestors = query_get_ancestors(store, child_id)
        return parent_id in ancestors
    except Exception as e:
        logging.warning(f"Error checking relationship between {parent_id} and {child_id}: {e}")
        return False


def _compile_patterns(patterns_dict):
    """Compile regex patterns for better performance."""
    return {name: re.compile(pattern) for name, pattern in patterns_dict.items()}


def _find_parent_disease(label, disease_id, label_to_id_map, compiled_patterns, store):
    """Find parent disease for a given label using pattern matching.

    Args:
        label: Disease label to match against patterns
        disease_id: Disease CURIE ID
        label_to_id_map: Dictionary mapping labels to IDs
        compiled_patterns: Compiled regex patterns
        store: PyOxigraph Store object for hierarchy queries

    Returns:
        Tuple of (parent_label, parent_id, pattern_name) or (None, None, None) if no match
    """
    for pattern_name, pattern in compiled_patterns.items():
        match = pattern.match(label)
        if not match:
            continue

        parent_label = match.group(1).strip().lower()
        parent_id = label_to_id_map.get(parent_label)

        if parent_id and _is_ancestor(store, parent_id, disease_id):
            return parent_label, parent_id, pattern_name

    return None, None, None


def _filter_mondo_labels(mondo_labels, chromosomal_diseases, human_diseases, mondo_prefix):
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


def _match_disease_subtypes(labels_df, compiled_patterns, store):
    """Match diseases to their parent groups using patterns.

    Args:
        labels_df: DataFrame with disease labels and IDs
        compiled_patterns: Compiled regex patterns for matching
        store: PyOxigraph Store object for hierarchy queries

    Returns:
        DataFrame with parent information added
    """
    label_to_id = dict(zip(labels_df["label_lower"], labels_df["disease_id"]))

    matches = labels_df.apply(
        lambda row: _find_parent_disease(row["label"], row["disease_id"], label_to_id, compiled_patterns, store),
        axis=1,
        result_type="expand",
    )
    matches.columns = ["parent_label", "parent_id", "pattern_name"]
    logger.info(f"Matched {matches['parent_id'].notna().sum()} diseases to parent groups ({len(matches)})")

    return pd.concat([labels_df, matches], axis=1)


def _build_subtype_counts(matched_df):
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


def _build_subtype_df(matched_df, mondo_subtype_subset, contributor):
    """Build dataframe from matched subtypes (no ROBOT headers).

    Returns DataFrame with columns: subject_id, subset_predicate, subset_object, contributor_predicate, contributor_object
    """
    unique_parents = matched_df[["parent_id", "parent_label"]].drop_duplicates().dropna().sort_values("parent_id")

    # Create rows for subset membership (one per parent)
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


def _log_mondo_size(mondo_graph):
    """Log the size of the MONDO graph in triples."""
    count_query = "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"
    result = list(mondo_graph.query(count_query))
    triple_count = result[0]["count"].value if result else 0
    logger.info(f"Working with MONDO graph containing {triple_count} triples")
    return triple_count


def extract_disease_data_from_mondo(
    mondo_graph,
    billable_icd10: pd.DataFrame,
    subtypes_params: Dict[str, Any],
    subtype_patterns: dict,
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

    _log_mondo_size(mondo_graph)

    # Step 0: Extract metadata and labels from MONDO
    logger.info("Step 0: Extracting metadata and labels from MONDO")

    # Extract labels (for internal use in subtype identification)
    mondo_labels = query_mondo_labels(mondo_graph)
    logger.info(f"Extracted {len(mondo_labels)} MONDO labels")

    # Extract metadata (primary output for documentation)
    mondo_metadata = query_ontology_metadata(mondo_graph)
    logger.info(f"Extracted MONDO metadata: {mondo_metadata}")

    # Extract obsoletes (primary output for documentation)
    mondo_obsoletes = query_mondo_obsoletes(mondo_graph)
    logger.info(f"Extracted {len(mondo_obsoletes)} obsolete MONDO terms")

    # Step 1: Identify disease subtypes using hierarchy analysis
    logger.info("Step 1: Identifying disease subtypes")

    subtype_counts, df_subtypes = _extract_subtype_data(mondo_graph, subtypes_params, subtype_patterns, mondo_labels)

    logger.info("Step 2: Enriching MONDO graph with annotations")

    _log_mondo_size(mondo_graph)

    logger.info("Step 3: Extracting disease data from enriched MONDO")
    disease_list_raw = query_raw_disease_list_data_from_mondo(mondo_graph, billable_icd10, df_subtypes)
    logger.info(f"Extracted {len(disease_list_raw)} diseases in raw list")

    # Run metrics query
    logger.info("Extracting disease metrics")
    mondo_metrics = query_matrix_disease_list_metrics(mondo_graph)
    logger.info(f"Extracted metrics for {len(mondo_metrics)} diseases")

    _log_mondo_size(mondo_graph)

    logger.info("Finished extracting all disease data from MONDO")
    return {
        "mondo_metadata": mondo_metadata,
        "mondo_obsoletes": mondo_obsoletes,
        "disease_list_raw": disease_list_raw,
        "mondo_metrics": mondo_metrics,
        "mondo_preprocessed": mondo_graph,
        "subtype_counts": subtype_counts,
    }


def _extract_subtype_data(mondo_graph, subtypes_params, subtype_patterns, mondo_labels):
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

    # Filter labels and match subtypes
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
def _matrix_disease_filter(df_disease_list_unfiltered):
    """
    The original matrix disease filtering function

    Parameters
    ----------
    df_disease_list_unfiltered : pandas.DataFrame
        The disease list, unfiltered, but with columns that are used for filtering.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        A tuple of two DataFrames: the first one contains the included diseases, and the second one contains the excluded diseases.
    """
    filter_column = "official_matrix_filter"

    # By default, no disease is included
    df_disease_list_unfiltered[filter_column] = False

    # QC: Check for conflicts where both f_matrix_manually_included and f_matrix_manually_excluded are True
    conflicts = df_disease_list_unfiltered[
        df_disease_list_unfiltered["f_matrix_manually_included"]
        & df_disease_list_unfiltered["f_matrix_manually_excluded"]
    ]

    if not conflicts.empty:
        # Format the conflicts nicely
        conflict_str = conflicts.to_string(index=False)
        raise ValueError(
            f"Conflicts found: The following entries are marked as both manually included and manually excluded:\n{conflict_str}"
        )

    # First, we add all manually curated classes to the list
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_matrix_manually_included"] == True

    # Next, we add all leaf classes
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_leaf"] == True

    # Now, we add all the immediate parents of leaf classes that are mapped to OMIM, ICD, or Orphanet
    df_disease_list_unfiltered[filter_column] |= (df_disease_list_unfiltered["f_leaf_direct_parent"] == True) & (
        (df_disease_list_unfiltered["f_omim"] == True)
        | (df_disease_list_unfiltered["f_omimps_descendant"] == True)
        | (df_disease_list_unfiltered["f_icd_category"] == True)
        | (df_disease_list_unfiltered["f_orphanet_disorder"] == True)
        | (df_disease_list_unfiltered["f_orphanet_subtype"] == True)
    )

    # Next, we add all diseases corresponding to ICD 10 billable codes
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_icd_category"] == True

    # Next, we add all diseases corresponding to Orphanet disorders
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_orphanet_disorder"] == True

    # Next, we add all diseases corresponding to ClinGen curated conditions
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_clingen"] == True

    # Next, we add all diseases corresponding to OMIM curated diseases
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered["f_omim"] == True

    # Next we remove all susceptibilities, mondo subtypes, and diseases with and/or or with/without
    # UPDATE 13.02.2025: We will for now _not_ remove these, but provide filter columns for them instead
    # df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_mondo_subtype'] == True, filter_column] = False
    # df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_susceptibility'] == True, filter_column] = False
    # df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_andor'] == True, filter_column] = False
    # df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_withorwithout'] == True, filter_column] = False

    ## Remove all hereditary diseases without classification. This is imo a dangerous default, but
    ## @jxneli reviewed all 849 cases from the February 2025 release and found that all were indeed
    ## "irrelevant" for drug repurposing, https://github.com/everycure-org/matrix-disease-list/issues/50
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered["f_unclassified_hereditary"] == True, filter_column] = (
        False
    )

    ## Remove all diseases that are candidates for obsoletion
    ## https://github.com/everycure-org/matrix-disease-list/issues/48
    # df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_obsoletion_candidate'] == True, filter_column] = False

    ## Remove all paraphilic disorders
    ## https://github.com/everycure-org/matrix-disease-list/issues/42
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered["f_paraphilic"] == True, filter_column] = False

    # Remove disease that were manually excluded
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered["f_matrix_manually_excluded"] == True, filter_column] = (
        False
    )

    return df_disease_list_unfiltered


def _extract_groupings(
    subsets: str, groupings: list, subset_prefix: str, subset_delimiter: str, grouping_delimiter: str
) -> dict:
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
    result = {grouping: [] for grouping in groupings}

    if subsets:
        for subset in subsets.split(subset_delimiter):
            subset = subset.strip()
            for grouping in groupings:
                if subset.startswith(f"{subset_prefix}{grouping}"):
                    # Extract tag by removing prefix, grouping name, spaces, and underscores
                    subset_tag = (
                        subset.replace(subset_prefix, "")
                        .replace(grouping, "")
                        .replace(" ", "")
                        .strip("_")
                        .replace(grouping_delimiter, "")
                    )
                    # Skip member tags and empty strings
                    if subset_tag and subset_tag != "member":
                        result[grouping].append(subset_tag)

    # Format results: join with pipe, exclude "other" if multiple values
    return {
        key: grouping_delimiter.join([v for v in values if v != "other"] if len(values) > 1 else values)
        if values
        else ""
        for key, values in result.items()
    }


def _is_grouping_heuristic(
    df: pd.DataFrame, grouping_columns: list, not_grouping_columns: list, output_column: str
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

    # Validate all expected columns are present
    expected_cols = set(grouping_columns) | set(not_grouping_columns)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing columns: {df.columns}")
        raise ValueError(f"DataFrame is missing expected grouping columns: {', '.join(sorted(missing))}")

    # Initialize to False
    df[output_column] = False

    # Apply positive grouping rules
    for col in grouping_columns:
        df[output_column] = df[output_column] | df[col].fillna(False)

    # Apply negative grouping rules (override)
    for col in not_grouping_columns:
        df.loc[df[col].fillna(False), output_column] = False

    return df


def _prepare_subtype_counts(subtype_counts: pd.DataFrame) -> pd.DataFrame:
    """Prepare subtype counts DataFrame for merging.

    Args:
        subtype_counts: DataFrame with subtype information

    Returns:
        Simplified DataFrame with category_class and count_subtypes columns
    """
    return (
        subtype_counts[["subset_group_id", "other_subsets_count"]]
        .drop_duplicates()
        .rename(columns={"subset_group_id": "category_class", "other_subsets_count": "count_subtypes"})
    )


def _extract_and_pivot_groupings(
    df: pd.DataFrame, groupings: list, subset_prefix: str, subset_delimiter: str, grouping_delimiter: str
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
    # Extract groupings from subsets column
    groupings_extracted = (
        df["subsets"]
        .apply(lambda x: _extract_groupings(x, groupings, subset_prefix, subset_delimiter, grouping_delimiter))
        .apply(pd.Series)
    )

    # Combine with category_class (drop label as not needed downstream)
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
    # Start with base and merge groupings
    merged = base_df.merge(groupings_df, on="category_class", how="left")

    # Merge metrics and subtype counts together first (both use category_class)
    metrics_and_counts = metrics_df.merge(subtype_counts_df, on="category_class", how="outer")
    merged = merged.merge(metrics_and_counts, on="category_class", how="left")

    # Finally merge full subtype info
    merged = merged.merge(
        full_subtype_counts[["subset_id", "subset_group_id", "subset_group_label", "other_subsets_count"]],
        left_on="category_class",
        right_on="subset_id",
        how="left",
    )

    # Clean up temporary column
    return merged.drop(columns=["subset_id"], errors="ignore")


def _normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to 'True'/'False' string format.

    Converts all columns starting with 'is_' and special case 'tag_existing_treatment'
    to string boolean values.

    Args:
        df: DataFrame with boolean columns

    Returns:
        DataFrame with normalized boolean strings
    """
    df = df.copy()

    # Find columns to convert
    columns_to_convert = [col for col in df.columns if col.startswith("is_")]

    # Special case for tag_existing_treatment
    # See: https://github.com/everycure-org/matrix-disease-list/issues/75
    if "tag_existing_treatment" in df.columns:
        columns_to_convert.append("tag_existing_treatment")

    # Convert to string booleans
    for col in columns_to_convert:
        df[col] = df[col].map(
            lambda x: "True" if (isinstance(x, bool) and x) or (isinstance(x, str) and x.lower() == "true") else "False"
        )

    return df


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
            - default_count: Default value for missing counts

    Returns:
        Dictionary with 'disease_list' key containing final DataFrame
    """
    logger.info("Create final disease list")

    # Extract parameters
    curated_groupings = parameters["curated_groupings"]
    llm_groupings = parameters["llm_groupings"]
    subset_prefix = parameters["subset_prefix"]
    subset_delimiter = parameters["subset_delimiter"]
    grouping_delimiter = parameters["grouping_delimiter"]
    grouping_columns = parameters["grouping_columns"]
    not_grouping_columns = parameters["not_grouping_columns"]
    grouping_heuristic_column = parameters["grouping_heuristic_column"]
    default_count = parameters["default_count"]

    # Stage 1: Prepare subtype counts
    subtype_group_counts = _prepare_subtype_counts(subtype_counts)

    # Stage 2: Apply MATRIX disease filtering
    filtered_df = _matrix_disease_filter(disease_list_raw)

    # Stage 3: Rename filter columns from f_ to is_ prefix
    # See: https://github.com/everycure-org/matrix-disease-list/issues/75
    filtered_df = filtered_df.rename(columns=lambda x: re.sub(r"^f_", "is_", x) if x.startswith("f_") else x)

    # Stage 4: Extract and pivot disease groupings
    all_groupings = curated_groupings + llm_groupings
    groupings_df = _extract_and_pivot_groupings(
        filtered_df[["category_class", "label", "subsets"]],
        all_groupings,
        subset_prefix,
        subset_delimiter,
        grouping_delimiter,
    )

    # Stage 5: Merge all data sources
    merged_df = _merge_disease_data_sources(
        filtered_df, groupings_df, mondo_metrics, subtype_group_counts, subtype_counts
    )

    # Stage 6: Compute derived columns
    # Convert count columns to numeric (PyOxigraph returns all values as strings)
    merged_df["count_subtypes"] = (
        pd.to_numeric(merged_df["count_subtypes"], errors="coerce").fillna(default_count).astype(int)
    )
    merged_df["count_descendants"] = (
        pd.to_numeric(merged_df["count_descendants"], errors="coerce").fillna(default_count).astype(int)
    )
    merged_df["count_descendants_without_subtypes"] = merged_df["count_descendants"] - merged_df["count_subtypes"]

    # Stage 7: Apply grouping heuristic
    merged_df = _is_grouping_heuristic(merged_df, grouping_columns, not_grouping_columns, grouping_heuristic_column)

    # Stage 8: Normalize boolean columns to string format
    final_df = _normalize_boolean_columns(merged_df)

    logger.info(f"Created disease list with {len(final_df)} entries")

    return {"disease_list": final_df}


def validate_disease_list(
    disease_list: pd.DataFrame,
) -> bool:
    """Validate the generated disease list.

    Args:
        disease_list: Final disease list

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating disease list")
    # length should be greater than 15000
    if len(disease_list) < 15000:
        raise ValueError(
            f"Validation failed: Disease list has only {len(disease_list)} entries, expected at least 15000"
        )

    # check for duplicate MONDO IDs
    if disease_list["category_class"].duplicated().any():
        duplicated_ids = disease_list[disease_list["category_class"].duplicated()]["category_class"].unique()
        raise ValueError(f"Validation failed: Duplicate MONDO IDs found: {duplicated_ids}")
