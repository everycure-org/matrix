"""Disease list pipeline nodes for MATRIX.

This pipeline processes the MONDO disease ontology to create the MATRIX disease list,
which defines which diseases are included in the drug repurposing platform.

The workflow is modeled after the matrix-disease-list repository:
https://github.com/everycure-org/matrix-disease-list
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from matrix.utils.system import run_subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
from oaklib.datamodels.vocabulary import IS_A


def _clean_sparql_results(df: pd.DataFrame) -> pd.DataFrame:
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


def _load_and_clean_sparql_dataframe(tsv_path: Path, clean: bool = True) -> pd.DataFrame:
    """Load a TSV file and optionally clean SPARQL results.

    Args:
        tsv_path: Path to the TSV file
        clean: Whether to apply SPARQL result cleaning (default: True)

    Returns:
        Loaded DataFrame, optionally cleaned
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, na_filter=False)
    if clean:
        df = _clean_sparql_results(df)
    return df


def extract_metadata_from_mondo(
    mondo_owl: str,
    ontology_metadata_query: str,
    mondo_obsoletes_query: str,
) -> Dict[str, Any]:
    """Extract labels, metadata, and obsoletes from the Mondo ontology.

    Args:
        mondo_owl: MONDO ontology content in OWL format
        ontology_metadata_query: Path to ontology metadata SPARQL query
        mondo_obsoletes_query: Path to mondo obsoletes SPARQL query

    Returns:
        Dictionary containing:
            - mondo_labels: DataFrame with disease IDs and labels
            - mondo_metadata: DataFrame with MONDO version info
            - mondo_obsoletes: DataFrame with obsolete terms
    """
    logger.info("Processing Mondo ontology: extracting labels, metadata, and obsoletes")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file once
        # TODO this probably needs to be done differently
        input_owl = tmpdir_path / "mondo.owl"
        with open(input_owl, "w") as f:
            f.write(mondo_owl)

        # Define output paths
        # TODO this probably needs to be done differently
        labels_tsv = tmpdir_path / "mondo_labels.tsv"
        metadata_tsv = tmpdir_path / "metadata.tsv"
        obsoletes_tsv = tmpdir_path / "obsoletes.tsv"

        # Run single ROBOT pipeline with chained commands
        cmd = f"""robot \
            export -i "{input_owl}" -f tsv --header "ID|LABEL" --export "{labels_tsv}" \
            query -f tsv \
                --query "{ontology_metadata_query}" "{metadata_tsv}" \
                --query "{mondo_obsoletes_query}" "{obsoletes_tsv}" """

        run_subprocess(cmd, check=True, stream_output=True)

        # Load and process results
        mondo_labels = pd.read_csv(labels_tsv, sep="\t")
        logger.info(f"Extracted {len(mondo_labels)} MONDO labels")

        mondo_metadata = _load_and_clean_sparql_dataframe(metadata_tsv)
        logger.info("Extracted MONDO metadata")

        mondo_obsoletes = _load_and_clean_sparql_dataframe(obsoletes_tsv)
        logger.info(f"Extracted {len(mondo_obsoletes)} obsolete MONDO terms")

        return {
            "mondo_labels": mondo_labels,
            "mondo_metadata": mondo_metadata,
            "mondo_obsoletes": mondo_obsoletes,
        }


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


def create_billable_icd10_template(
    icd10_codes: pd.DataFrame,
    mondo_sssom: pd.DataFrame,
    exact_match_predicate: str,
    icd10_billable_subset: str,
    subset_annotation: str,
    icd10cm_prefix: str,
) -> pd.DataFrame:
    """Create ROBOT template for billable ICD-10-CM codes mapped to MONDO.

    Args:
        icd10_codes: DataFrame with ICD-10-CM codes (must have 'CODE' column)
        mondo_sssom: MONDO SSSOM mappings (must have 'predicate_id', 'object_id', 'subject_id')
        exact_match_predicate: SKOS predicate for exact matches (e.g., "skos:exactMatch")
        icd10_billable_subset: URI for billable ICD-10 subset
        subset_annotation: ROBOT template annotation for subset membership
        icd10cm_prefix: CURIE prefix for ICD-10-CM codes (e.g., "ICD10CM")

    Returns:
        ROBOT template DataFrame with billable ICD-10 codes
    """
    logger.info("Creating billable ICD-10-CM template")

    # Convert ICD-10 codes to CURIE format without mutating input
    icd10_codes_formatted = icd10_codes.copy()
    icd10_codes_formatted["CODE"] = icd10_codes_formatted["CODE"].apply(
        lambda code: _format_icd10_code_to_curie(code, icd10cm_prefix)
    )

    # Filter SSSOM mappings to only exact matches
    exact_matches = mondo_sssom[mondo_sssom["predicate_id"] == exact_match_predicate].copy()

    # Find billable codes that have MONDO mappings
    billable_with_mappings = icd10_codes_formatted[
        icd10_codes_formatted["CODE"].isin(exact_matches["object_id"])
    ]

    logger.info(f"Found {len(billable_with_mappings)} billable ICD-10 codes with MONDO mappings")

    # Create template rows by joining billable codes with their MONDO mappings
    template_data = exact_matches.merge(
        billable_with_mappings[["CODE"]], left_on="object_id", right_on="CODE", how="inner"
    )

    # Build ROBOT template with required columns
    robot_template = pd.DataFrame(
        {
            "ID": template_data["subject_id"],
            "SUBSET": icd10_billable_subset,
            "ICD10CM_CODE": template_data["object_id"],
        }
    )

    # Add ROBOT template header row
    header = pd.DataFrame({"ID": ["ID"], "SUBSET": [subset_annotation], "ICD10CM_CODE": [""]})

    return pd.concat([header, robot_template], ignore_index=True)


def _is_parent(mondo, parent_id, child_id):
    """Check if parent_id is an ancestor of child_id in the ontology."""
    try:
        parents = mondo.ancestors([child_id], predicates=[IS_A])
        return parent_id in parents
    except Exception as e:
        logging.warning(f"Error checking relationship between {parent_id} and {child_id}: {e}")
        return False


def _compile_patterns(patterns_dict):
    """Compile regex patterns for better performance."""
    return {name: re.compile(pattern) for name, pattern in patterns_dict.items()}


def _find_parent_disease(label, disease_id, label_to_id_map, compiled_patterns, mondo):
    """Find parent disease for a given label using pattern matching.

    Returns:
        Tuple of (parent_label, parent_id, pattern_name) or (None, None, None) if no match
    """
    for pattern_name, pattern in compiled_patterns.items():
        match = pattern.match(label)
        if not match:
            continue

        parent_label = match.group(1).strip().lower()
        parent_id = label_to_id_map.get(parent_label)

        if parent_id and _is_parent(mondo, parent_id, disease_id):
            return parent_label, parent_id, pattern_name

    return None, None, None


def _filter_mondo_labels(mondo_labels, chromosomal_diseases, human_diseases, mondo_prefix):
    """Filter and prepare MONDO labels for subtype matching."""
    filtered = (
        mondo_labels
        .dropna(subset=["LABEL"])
        .query(f"ID.str.startswith('{mondo_prefix}')")
        .loc[lambda df: ~df["ID"].isin(chromosomal_diseases)]
        .loc[lambda df: df["ID"].isin(human_diseases)]
        [["ID", "LABEL"]]
        .rename(columns={"ID": "disease_id", "LABEL": "label"})
    )
    filtered["label_lower"] = filtered["label"].str.lower()
    return filtered


def _match_disease_subtypes(labels_df, compiled_patterns, mondo):
    """Match diseases to their parent groups using patterns."""
    label_to_id = dict(zip(labels_df["label_lower"], labels_df["disease_id"]))

    matches = labels_df.apply(
        lambda row: _find_parent_disease(
            row["label"], row["disease_id"], label_to_id, compiled_patterns, mondo
        ),
        axis=1,
        result_type="expand"
    )
    matches.columns = ["parent_label", "parent_id", "pattern_name"]

    return pd.concat([labels_df, matches], axis=1)


def _build_subtype_counts(matched_df):
    """Build counts of subtypes per parent disease."""
    valid_matches = matched_df.dropna(subset=["parent_label"])
    valid_matches = valid_matches[valid_matches["parent_label"].str.strip() != ""]

    counts = (
        valid_matches
        .groupby("parent_label")
        .size()
        .reset_index(name="count")
    )

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


def _build_robot_template(matched_df, mondo_subtype_subset, contributor, subset_annotation, contributor_annotation):
    """Build ROBOT template from matched subtypes."""
    unique_parents = (
        matched_df[["parent_id", "parent_label"]]
        .drop_duplicates()
        .dropna()
        .sort_values("parent_id")
        .rename(columns={"parent_id": "ID", "parent_label": "LABEL"})
    )

    unique_parents["SUBSET"] = mondo_subtype_subset
    unique_parents["CONTRIBUTOR"] = contributor

    header = pd.DataFrame({
        "ID": ["ID"],
        "LABEL": [""],
        "SUBSET": [subset_annotation],
        "CONTRIBUTOR": [contributor_annotation],
    })

    template = pd.concat([header, unique_parents], ignore_index=True)
    return template.sort_values("ID", ignore_index=True)


def create_subtypes_template(
    mondo_labels: pd.DataFrame,
    mondo_owl: str,
    chromosomal_diseases_root: str,
    human_diseases_root: str,
    chromosomal_diseases_exceptions: list,
    subtype_patterns: dict,
    mondo_prefix: str,
    mondo_subtype_subset: str,
    default_contributor: str,
    subset_annotation_split: str,
    contributor_annotation: str,
) -> dict:
    """Create template with high granularity disease subtypes.

    Args:
        mondo_labels: DataFrame with MONDO labels (must have 'ID' and 'LABEL' columns)
        mondo_owl: MONDO ontology content in OWL format
        chromosomal_diseases_root: Root MONDO ID for chromosomal diseases to exclude
        human_diseases_root: Root MONDO ID for human diseases to include
        chromosomal_diseases_exceptions: List of chromosomal disease IDs that should NOT be excluded
        subtype_patterns: Dictionary of regex patterns for matching disease subtypes
        mondo_prefix: CURIE prefix for MONDO IDs (e.g., "MONDO:")
        mondo_subtype_subset: URI for MONDO subtype subset
        default_contributor: ORCID URI for contributor attribution
        subset_annotation_split: ROBOT template annotation for subset with splitting
        contributor_annotation: ROBOT template annotation for contributor

    Returns:
        Dictionary containing:
            - subtypes_template: ROBOT template DataFrame
            - subtypes_counts: DataFrame with subtype counts per parent
    """
    logger.info("Creating subtypes template")
    from oaklib import get_adapter

    # TODO probably wrong: Write MONDO OWL content to temporary file for OAK to read
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False)
    tmp_file.write(mondo_owl)
    tmp_file.flush()
    tmp_file.close()
    mondo_owl_path = tmp_file.name

    try:
        # Load ontology and get disease hierarchies
        mondo = get_adapter(f"pronto:{mondo_owl_path}")

        chromosomal_diseases = set(mondo.descendants([chromosomal_diseases_root], predicates=[IS_A]))
        human_diseases = set(mondo.descendants([human_diseases_root], predicates=[IS_A]))

        # Remove exceptions from chromosomal disease exclusion list
        for exception_id in chromosomal_diseases_exceptions:
            chromosomal_diseases.discard(exception_id)

        # Filter labels to relevant human diseases
        filtered_labels = _filter_mondo_labels(
            mondo_labels, chromosomal_diseases, human_diseases, mondo_prefix
        )

        # Compile patterns for performance
        compiled_patterns = _compile_patterns(subtype_patterns)

        # Match diseases to their parent groups
        matched = _match_disease_subtypes(filtered_labels, compiled_patterns, mondo)

        # Build count information
        subtype_counts = _build_subtype_counts(matched)

        # Build ROBOT template
        robot_template = _build_robot_template(
            matched, mondo_subtype_subset, default_contributor,
            subset_annotation_split, contributor_annotation
        )

        logger.info(f"Identified {len(subtype_counts)} disease subtypes")

        return {
            "subtypes_template": robot_template,
            "subtypes_counts": subtype_counts,
        }
    finally:
        # Clean up temporary file
        Path(mondo_owl_path).unlink(missing_ok=True)


def _build_robot_command(
    input_owl: Path,
    billable_template: Path,
    subtypes_template: Path,
    update_queries: list,
    select_queries: dict,
    output_owl: Path,
) -> str:
    """Build ROBOT command for merging templates and extracting data.

    Args:
        input_owl: Path to input OWL file
        billable_template: Path to billable ICD-10 template
        subtypes_template: Path to subtypes template
        update_queries: List of paths to SPARQL UPDATE queries
        select_queries: Dict mapping query paths to output TSV paths
        output_owl: Path for output preprocessed OWL file

    Returns:
        ROBOT command string
    """
    # Build template merge command
    cmd_parts = [
        "robot",
        f'template -i "{input_owl}"',
        "--merge-after",
        f'--template "{billable_template}"',
        f'--template "{subtypes_template}"',
    ]

    # Add UPDATE queries
    for query_path in update_queries:
        cmd_parts.append(f'query --update "{query_path}"')

    # Add SELECT queries
    if select_queries:
        cmd_parts.append("query -f tsv")
        for query_path, output_path in select_queries.items():
            cmd_parts.append(f'--query "{query_path}" "{output_path}"')

    # Add final merge output
    cmd_parts.append(f'merge -o "{output_owl}"')

    return " \\\n    ".join(cmd_parts)


def process_mondo_with_templates(
    mondo_owl: str,
    billable_icd10: pd.DataFrame,
    subtypes: pd.DataFrame,
    inject_mondo_top_grouping_query: str,
    inject_susceptibility_subset_query: str,
    inject_subset_declaration_query: str,
    downfill_disease_groupings_query: str,
    disease_groupings_other_query: str,
    disease_list_filters_query: str,
    metrics_query: str,
) -> Dict[str, Any]:
    """Merge templates into MONDO and extract disease list data.

    Args:
        mondo_owl: Base MONDO ontology content
        billable_icd10: Billable ICD-10 template DataFrame
        subtypes: Subtypes template DataFrame
        inject_mondo_top_grouping_query: Path to inject top grouping UPDATE query
        inject_susceptibility_subset_query: Path to inject susceptibility UPDATE query
        inject_subset_declaration_query: Path to inject subset declaration UPDATE query
        downfill_disease_groupings_query: Path to downfill groupings UPDATE query
        disease_groupings_other_query: Path to disease groupings other UPDATE query
        disease_list_filters_query: Path to disease list filters SELECT query
        metrics_query: Path to metrics SELECT query

    Returns:
        Dictionary containing:
            - disease_list_raw: DataFrame with raw disease list features
            - mondo_metrics: DataFrame with disease metrics
            - mondo_preprocessed: Preprocessed MONDO ontology content (for downstream use)
    """
    logger.info("Processing MONDO with templates and extracting data in single ROBOT pipeline")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # TODO Fix: Write MONDO OWL content to temporary file
        mondo_path = tmpdir_path / "mondo.owl"
        with open(mondo_path, "w") as f:
            f.write(mondo_owl)

        # Write DataFrames to temporary template files
        billable_path = tmpdir_path / "billable.robot.tsv"
        subtypes_path = tmpdir_path / "subtypes.robot.tsv"
        billable_icd10.to_csv(billable_path, sep="\t", index=False)
        subtypes.to_csv(subtypes_path, sep="\t", index=False)

        # Define output paths
        preprocessed_path = tmpdir_path / "mondo-with-subsets.owl"
        disease_list_tsv = tmpdir_path / "disease-list-raw.tsv"
        metrics_tsv = tmpdir_path / "mondo-metrics.tsv"

        # Build ROBOT command
        update_queries = [
            inject_mondo_top_grouping_query,
            inject_susceptibility_subset_query,
            inject_subset_declaration_query,
            downfill_disease_groupings_query,
            disease_groupings_other_query,
        ]

        select_queries = {
            disease_list_filters_query: disease_list_tsv,
            metrics_query: metrics_tsv,
        }

        cmd = _build_robot_command(
            mondo_path, billable_path, subtypes_path,
            update_queries, select_queries, preprocessed_path
        )

        logger.info("Running ROBOT pipeline")
        run_subprocess(cmd, check=True, stream_output=True)

        # Load preprocessed ontology
        with open(preprocessed_path, "r") as f:
            mondo_preprocessed = f.read()
        logger.info("Successfully merged templates into MONDO ontology")

        # Load and post-process results
        disease_list_raw = _load_and_clean_sparql_dataframe(disease_list_tsv)
        logger.info(f"Extracted {len(disease_list_raw)} diseases in raw list")

        mondo_metrics = _load_and_clean_sparql_dataframe(metrics_tsv)
        logger.info(f"Extracted metrics for {len(mondo_metrics)} diseases")

        return {
            "disease_list_raw": disease_list_raw,
            "mondo_metrics": mondo_metrics,
            "mondo_preprocessed": mondo_preprocessed,
        }


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


def _extract_groupings(subsets: str, groupings: list, subset_prefix: str, subset_delimiter: str, grouping_delimiter: str) -> dict:
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
                        subset
                        .replace(subset_prefix, "")
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
        key: grouping_delimiter.join(
            [v for v in values if v != "other"] if len(values) > 1 else values
        ) if values else ""
        for key, values in result.items()
    }


def _is_grouping_heuristic(
    df: pd.DataFrame,
    grouping_columns: list,
    not_grouping_columns: list,
    output_column: str
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
    df: pd.DataFrame,
    groupings: list,
    subset_prefix: str,
    subset_delimiter: str,
    grouping_delimiter: str
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
    groupings_extracted = df["subsets"].apply(
        lambda x: _extract_groupings(x, groupings, subset_prefix, subset_delimiter, grouping_delimiter)
    ).apply(pd.Series)

    # Combine with category_class (drop label as not needed downstream)
    result = pd.concat([df[["category_class"]], groupings_extracted], axis=1)
    return result.sort_values("category_class")


def _merge_disease_data_sources(
    base_df: pd.DataFrame,
    groupings_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    subtype_counts_df: pd.DataFrame,
    full_subtype_counts: pd.DataFrame
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
        how="left"
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
    curated_groupings: list,
    llm_groupings: list,
    subset_prefix: str,
    subset_delimiter: str,
    grouping_delimiter: str,
    grouping_columns: list,
    not_grouping_columns: list,
    grouping_heuristic_column: str,
    default_count: int,
) -> Dict[str, pd.DataFrame]:
    """Apply filters and transformations to create final disease list.

    Args:
        disease_list_raw: Raw disease list with filter features
        mondo_metrics: MONDO metrics DataFrame
        subtype_counts: Subtype counts and relationships
        curated_groupings: List of curated disease grouping names
        llm_groupings: List of LLM-generated grouping names
        subset_prefix: Prefix for subset parsing (e.g., "mondo:")
        subset_delimiter: Delimiter for subset strings (e.g., ";")
        grouping_delimiter: Delimiter for joining grouping values (e.g., "|")
        grouping_columns: Columns indicating a disease is a grouping
        not_grouping_columns: Columns that override grouping classification
        grouping_heuristic_column: Name of output heuristic column
        default_count: Default value for missing counts

    Returns:
        Dictionary with 'disease_list' key containing final DataFrame
    """
    logger.info("Create final disease list")

    # Stage 1: Prepare subtype counts
    subtype_group_counts = _prepare_subtype_counts(subtype_counts)

    # Stage 2: Apply MATRIX disease filtering
    filtered_df = _matrix_disease_filter(disease_list_raw)

    # Stage 3: Rename filter columns from f_ to is_ prefix
    # See: https://github.com/everycure-org/matrix-disease-list/issues/75
    filtered_df = filtered_df.rename(
        columns=lambda x: re.sub(r"^f_", "is_", x) if x.startswith("f_") else x
    )

    # Stage 4: Extract and pivot disease groupings
    all_groupings = curated_groupings + llm_groupings
    groupings_df = _extract_and_pivot_groupings(
        filtered_df[["category_class", "label", "subsets"]],
        all_groupings,
        subset_prefix,
        subset_delimiter,
        grouping_delimiter
    )

    # Stage 5: Merge all data sources
    merged_df = _merge_disease_data_sources(
        filtered_df,
        groupings_df,
        mondo_metrics,
        subtype_group_counts,
        subtype_counts
    )

    # Stage 6: Compute derived columns
    merged_df["count_subtypes"] = merged_df["count_subtypes"].fillna(default_count)
    merged_df["count_descendants"] = merged_df["count_descendants"].fillna(default_count)
    merged_df["count_descendants_without_subtypes"] = (
        merged_df["count_descendants"] - merged_df["count_subtypes"]
    )

    # Stage 7: Apply grouping heuristic
    merged_df = _is_grouping_heuristic(
        merged_df,
        grouping_columns,
        not_grouping_columns,
        grouping_heuristic_column
    )

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
