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
from typing import Any, Dict, List, Optional

import pandas as pd

from matrix.utils.system import run_subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
from oaklib.datamodels.vocabulary import IS_A


def extract_mondo_labels(mondo_owl: str) -> pd.DataFrame:
    """Extract disease labels from MONDO ontology using ROBOT.

    Args:
        mondo_owl: MONDO ontology content in OWL format

    Returns:
        DataFrame with disease IDs and labels
    """
    logger.info("Extracting MONDO labels from ontology")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file
        input_owl = tmpdir_path / "mondo.owl"
        output_tsv = tmpdir_path / "mondo_labels.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_owl)

        # Run ROBOT export command to extract ID and LABEL columns
        cmd = f'robot export -i "{input_owl}" -f tsv --header "ID|LABEL" --export "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the exported TSV
        df = pd.read_csv(output_tsv, sep="\t")
        logger.info(f"Extracted {len(df)} MONDO labels")

        return df


def create_billable_icd10_template(
    icd10_codes: pd.DataFrame,
    mondo_sssom: pd.DataFrame,
) -> pd.DataFrame:
    """Create a list (formatted as a ROBOT template) for billable ICD-10-CM codes mapped to MONDO.

    Args:
        icd10_codes: ICD-10-CM codes file in XLSX format
        mondo_sssom: MONDO SSSOM mappings

    Returns:
        DataFrame with billable ICD-10 codes template
    """
    logger.info("Creating billable ICD-10-CM template")

    # Transform ICD-10 codes in the first column to CURIE format
    # Example: "A001" -> "ICD10CM:A00.1"
    icd10_codes.iloc[:, 0] = icd10_codes.iloc[:, 0].apply(
        lambda x: f"ICD10CM:{x[:3]}.{x[3:]}" if pd.notna(x) and len(x) > 3 else x
    )

    # Extract ICD-10 codes that have exactMatch in MONDO SSSOM
    icd10_codes_in_mondo = mondo_sssom.loc[
        mondo_sssom['predicate_id'] == 'skos:exactMatch', 'object_id'
    ].unique()

    # Filter to only billable codes that are in MONDO
    icd10_codes_billable = icd10_codes[icd10_codes.iloc[:, 0].isin(icd10_codes_in_mondo)]["CODE"].unique()

    logger.info(f"Found {len(icd10_codes_billable)} billable ICD-10 codes with MONDO mappings")

    rows = []

    for icd10_code in icd10_codes_billable:
        # Find MONDO IDs that have exactMatch to this ICD-10 code
        row_to_add = mondo_sssom[
            (mondo_sssom['predicate_id'] == 'skos:exactMatch') &
            (mondo_sssom['object_id'] == icd10_code)
        ][['subject_id', 'object_id']].copy()

        row_to_add['SUBSET'] = "http://purl.obolibrary.org/obo/mondo#icd10_billable"
        row_to_add['ID'] = row_to_add['subject_id']
        row_to_add['ICD10CM_CODE'] = row_to_add['object_id']
        rows.append(row_to_add[['ID', 'SUBSET', 'ICD10CM_CODE']])
    
    robot_template_header = pd.DataFrame({
        'ID': ['ID'],
        'SUBSET': ['AI oboInOwl:inSubset'],
        'ICD10CM_CODE': [""]
        })
    
    df_icd10billable_subsets = pd.concat(rows)
    df_icd10billable_subsets = pd.concat([robot_template_header, df_icd10billable_subsets]).reset_index(drop=True)

    # Write the filtered DataFrame to a TSV file
    return df_icd10billable_subsets


def _is_parent(mondo, parent_id, child_id):
    try:
        parents = mondo.ancestors([child_id], predicates=[IS_A])  # get all superclasses
        return parent_id in parents
    except Exception as e:
        logging.warning(f"Error checking relationship between {parent_id} and {child_id}: {e}")
        return False


def match_patterns_efficiently(df_labels, i_labels, patterns, mondo):
    label_match = []
    curie_match = []
    label_pattern = []
    
    for _, row in df_labels.iterrows():
        label = row["label"]
        disease_id = row["category_class"]
        match_found = False

        for pattern_id, pattern in patterns.items():
            match = re.match(pattern, label)
            
            if match:
                potential_disease_group_label = match.group(1)
            else:
                potential_disease_group_label = None
            
            if potential_disease_group_label:
                potential_disease_group_label = potential_disease_group_label.strip().lower()
                if potential_disease_group_label in i_labels:
                    potential_disease_group_id = i_labels[potential_disease_group_label]
                    if _is_parent(mondo, potential_disease_group_id, disease_id):
                        match_found = True
                        label_match.append(potential_disease_group_label)
                        label_pattern.append(pattern_id)
                        curie_match.append(potential_disease_group_id)
                        break

        if not match_found:
            label_match.append(None)
            label_pattern.append(None)
            curie_match.append(None)

    # Add the results to the DataFrame
    df_labels["label_match"] = label_match
    df_labels["label_pattern"] = label_pattern
    df_labels["curie_match"] = curie_match
    return df_labels


def create_subtypes_template(
    mondo_labels: pd.DataFrame,
    mondo_owl: str,
) -> pd.DataFrame:
    """Create template with high granularity disease subtypes.

    Args:
        mondo_labels: DataFrame with MONDO labels
        mondo_owl: MONDO ontology

    Returns:
        DataFrame with subtype template
    """
    logger.info("Creating subtypes template")
    from oaklib import get_adapter
    from oaklib.datamodels.vocabulary import IS_A

    # Write MONDO OWL content to temporary file for OAK to read
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False)
    tmp_file.write(mondo_owl)
    tmp_file.flush()
    tmp_file.close()
    mondo_owl_path = tmp_file.name

    try:
        oak_adapter = "pronto:{}".format(mondo_owl_path)
        mondo = get_adapter(oak_adapter)

        # We will exclude chromosomal diseases from the subtype process
        # as they are usually subtyped by chromosomal location
        # making them very different diseases
        chromosomal_diseases = set(mondo.descendants(["MONDO:0019040"], predicates=[IS_A]))
        human_diseases = set(mondo.descendants(["MONDO:0700096"], predicates=[IS_A]))

        # Some chromosomal diseases are indeed part of a series so we manually remove them
        chromosomal_diseases.remove("MONDO:0010767")
        chromosomal_diseases.remove("MONDO:0010763")

        # Load the data
        mondo_labels = mondo_labels.dropna(subset=['LABEL'])
        mondo_labels = mondo_labels[mondo_labels['ID'].str.startswith('MONDO:')]
        mondo_labels = mondo_labels[~mondo_labels['ID'].isin(chromosomal_diseases)]
        mondo_labels = mondo_labels[mondo_labels['ID'].isin(human_diseases)]
        mondo_labels = mondo_labels[["ID", "LABEL"]]
        mondo_labels.columns = ["category_class", "label"]
        mondo_labels['label_lc'] = mondo_labels['label'].str.lower()
        i_labels = {row['label_lc']: row['category_class'] for _, row in mondo_labels.iterrows()}

        # Define patterns
        patterns = {
        "autosomal_rd_o_x_d": r"autosomal[ ](?:recessive|dominant)[ ](?:juvenile|early[-]onset)(.*)[ ][0-9]+[0-9A-Za-z]*$",
        "x_typec_ad": r"(.*)[,][ ]type[ ][A-Z0-9]+$",
        "x_type_I": r"(.*)\s+type\s+(X?(IX|IV|V?I{1,3}))$",
        "x_type_ad": r"(.*)[ ]type[ ][A-Z0-9]+$",
        "x_group_a": r"(.*)[ ]group [A-Z]+$",
        "x_xylinked_d": r"(.*),[ ][xyXY][-]linked,[ ][0-9]+$",
        "x_xylinked": r"(.*),[ ][xyXY][-]linked$",
        "autosomal_rd_x_d": r"autosomal[ ](?:recessive|dominant)[ ](.*)[ ][0-9]+[0-9A-Za-z]*$",
        "x_variant_type": r"(.*)[ ]variant[ ]type$",
        "x_autosomomal_dominant_mild": r"(.*),[ ]autosomal[ ]dominant,[ ]mild$",
        "x_d_autosomomal_dominant": r"(.*) [0-9]+[0-9A-Za-z]*,[ ]autosomal[ ]dominant$",
        "x_d_early_onset": r"(.*) [0-9]+[0-9A-Za-z]*,[ ]early[- ]onset$",
        "x_mitochondrial": r"(.*),[ ]mitochondrial$",
        "x_xylinked": r"(.*),[ ][xyXY][-]linked$",
        "x_familial_d": r"(.*),[ ]familial,[ ][0-9]+$",
        "x_autosomal_rd_d": r"(.*),[ ]autosomal[ ](?:recessive|dominant),[ ][0-9]+$",
        "x_dueto_d": r"(.*)[,]?[ ]due[ ]to[ ].*$",
        "x_dp": r"(.*)[ ][0-9]+[qp]$",
        "x_d": r"(.*)[ ][0-9]+[0-9A-Za-z]*$",
        "x_tda": r"(.*)[ ]type[ ][0-9]+[A-Za-z0-9/_, ()-]+$",
        "x_d_ca": r"(.*)[ ][0-9]+[0-9A-Za-z]*[,][ ][A-Za-z0-9/_, ()-]+$",
        "x_d_a": r"(.*)[ ][0-9]+[0-9A-Za-z]*[ ][A-Za-z0-9/_, ()-]+$",
        "x_da_a": r"(.*)[ ][0-9]+[a-z]*[ ][A-Za-z0-9/_, ()-]+$",
        "x_I": r"(.*)\s(X?(?:IX|IV|V?I{1,3}))$",
        "x_a": r"(.*)[ ][A-Z]+$",
        "x_a_type": r"(.*)[,][ ][A-Za-z0-9-_]+[ ]type$",
        "familial_x": r"familial[ ](.*)$",
        "paroxysmal_x": r"paroxysmal[ ](.*)$",
        "persistent_x": r"persistent[ ](.*)$",
        "onset_x": r"(?:young|late|juvenile|early)[- ]onset[ ](.*)$",
        "onset_x_d": r"(?:young|late|juvenile|early)[- ]onset[ ](.*)[ ][0-9]+[0-9A-Za-z]*$",
        "persistent_x": r"persistent[ ](.*)$",
            "xylinked_x": r"[xyXY][-]linked[ ](.*)$",
        }

        df_disease_list_matched = match_patterns_efficiently(mondo_labels, i_labels, patterns, mondo)

        df_disease_list_matched_subset_with_matched_label_ids = df_disease_list_matched[["category_class", "label", "label_match", "curie_match", "label_pattern"]]
        df_disease_list_matched_subset_with_matched_label_ids.sort_values(by="label_match", inplace=True)

        # Count the number of subtypes for each disease
        df_disease_list_matched_subset_with_matched_label_ids = df_disease_list_matched_subset_with_matched_label_ids[
            df_disease_list_matched_subset_with_matched_label_ids['label_match'].notna() &
            (df_disease_list_matched_subset_with_matched_label_ids['label_match'].str.strip() != "")
        ]
        grouped_data_label_match = df_disease_list_matched_subset_with_matched_label_ids.groupby(["label_match"]).size()
        grouped_df_label_match = grouped_data_label_match.reset_index(name="count")

        # TODO consider dropping this QC step?
        processed = df_disease_list_matched_subset_with_matched_label_ids['label_pattern'].unique()
        for label_pattern in patterns.keys():
            if label_pattern not in processed:
                logging.warning(f"Label pattern {label_pattern} not in patterns.keys()")

        # Get the subset of the DataFrame that matches the top groupings
        top_subset_df = df_disease_list_matched_subset_with_matched_label_ids.copy()

        top_subset_df = pd.merge(top_subset_df, grouped_df_label_match, left_on="label_match", right_on='label_match', how="left")
        top_subset_df_out=top_subset_df[["category_class", "label","curie_match", "label_match","count"]]
        top_subset_df_out.columns=["subset_id", "subset_label", "subset_group_id", "subset_group_label", "other_subsets_count"]

        # Display the final filtered DataFrame
        final_subset_df = top_subset_df[["curie_match", "label_match"]].drop_duplicates().sort_values(by="curie_match")
        final_subset_df['subset'] = "http://purl.obolibrary.org/obo/mondo#mondo_subtype"
        final_subset_df['contributor'] = "https://orcid.org/0000-0002-7356-1779"
        final_subset_df.columns=["ID", "LABEL", "SUBSET", "CONTRIBUTOR"]

        robot_template_header = pd.DataFrame({
            'ID': ['ID'],
            'LABEL': [''],
            'SUBSET': ['AI oboInOwl:inSubset SPLIT=|'],
            'CONTRIBUTOR': ['>AI dc:contributor SPLIT=|'],
            })

        output_df = pd.concat([robot_template_header, final_subset_df]).reset_index(drop=True)
        output_df.sort_values(by='ID', inplace=True)

        return {
            "subtypes_template": output_df,
            "subtypes_counts": top_subset_df_out,
        }
    finally:
        # Clean up temporary file
        Path(mondo_owl_path).unlink(missing_ok=True)


def merge_templates_into_mondo(
    mondo_owl: str,
    billable_icd10: pd.DataFrame,
    subtypes: pd.DataFrame,
) -> str:
    """Merge all disease templates into MONDO ontology using ROBOT.

    This function mimics the Makefile logic for mondo-with-manually-curated-subsets.owl:
    - Merges multiple ROBOT templates into MONDO
    - Runs SPARQL update queries to inject subsets and groupings
    - Annotates and converts the final ontology

    Args:
        mondo_owl: Path to base MONDO ontology
        billable_icd10: Billable ICD-10 template
        subtypes: Subtypes template

    Returns:
        Path to modified MONDO ontology with subset annotations
    """
    logger.info("Merging disease templates into MONDO ontology")

    # Create temporary files for templates
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write MONDO OWL content to temporary file
        mondo_path = tmpdir_path / "mondo.owl"
        with open(mondo_path, "w") as f:
            f.write(mondo_owl)

        # Write DataFrames to temporary template files
        billable_path = tmpdir_path / "billable.robot.tsv"
        subtypes_path = tmpdir_path / "subtypes.robot.tsv"
        billable_icd10.to_csv(billable_path, sep="\t", index=False)
        subtypes.to_csv(subtypes_path, sep="\t", index=False)

        # Build ROBOT command to merge all templates
        output_path = tmpdir_path / "mondo-with-subsets.owl"

        # Get the queries directory path
        queries_dir = Path(__file__).parent / "queries"

        cmd = f"""robot \
            template -i "{mondo_path}" \
            --merge-after \
            --template "{billable_path}" \
            --template "{subtypes_path}" \
            query --update "{queries_dir / "inject-mondo-top-grouping.ru"}" \
            query --update "{queries_dir / "inject-susceptibility-subset.ru"}" \
            query --update "{queries_dir / "inject-subset-declaration.ru"}" \
            query --update "{queries_dir / "downfill-disease-groupings.ru"}" \
            query --update "{queries_dir / "disease-groupings-other.ru"}" \
            -o "{output_path}" """

        logger.info("Running ROBOT template merge command")
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output file and return its contents as a string
        # For now, we'll return the path - the catalog should handle this as text.TextDataset
        with open(output_path, "r") as f:
            owl_content = f.read()

        logger.info("Successfully merged templates into MONDO ontology")
        return owl_content


def extract_disease_list_raw(
    mondo_preprocessed: str,
    sparql_query_path: str,
) -> pd.DataFrame:
    """Extract raw features of the disease list from Mondo using ROBOT query.

    Args:
        mondo_preprocessed: MONDO ontology content with subset annotations
        sparql_query_path: Path to SPARQL query file (matrix-disease-list-filters.sparql)

    Returns:
        DataFrame with raw disease list features
    """
    logger.info("Extracting raw disease list features")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file
        input_owl = tmpdir_path / "mondo-with-subsets.owl"
        output_tsv = tmpdir_path / "disease-list-raw.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_preprocessed)

        # Run ROBOT query command
        cmd = f'robot query -i "{input_owl}" -f tsv --query "{sparql_query_path}" "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output TSV with dtype specification to handle mixed types
        df = pd.read_csv(output_tsv, sep="\t", dtype=str, na_filter=False)

        # Post-process to clean up SPARQL output
        # Remove ? prefix, angle brackets, and clean up IRIs
        for col in df.columns:
            df[col] = (
                df[col]
                .str.replace("?", "", regex=False)
                .str.replace("<http://purl.obolibrary.org/obo/MONDO_", "MONDO:", regex=False)
                .str.replace("http://purl.obolibrary.org/obo/mondo#", "mondo:", regex=False)
                .str.replace(">", "", regex=False)
            )
        df.columns = [col.replace("?", "") for col in df.columns]
        logger.info(f"Extracted {len(df)} diseases in raw list")
        return df


def extract_mondo_metrics(
    mondo_preprocessed: str,
    sparql_query_path: str,
) -> pd.DataFrame:
    """Extract metrics for disease list filtering using ROBOT query.

    Args:
        mondo_preprocessed: MONDO ontology content with subset annotations
        sparql_query_path: Path to SPARQL query file (matrix-disease-list-metrics.sparql)

    Returns:
        DataFrame with disease metrics
    """
    logger.info("Extracting disease metrics")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file
        input_owl = tmpdir_path / "mondo-with-subsets.owl"
        output_tsv = tmpdir_path / "mondo-metrics.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_preprocessed)

        # Run ROBOT query command
        cmd = f'robot query -i "{input_owl}" -f tsv --query "{sparql_query_path}" "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output TSV
        df = pd.read_csv(output_tsv, sep="\t")

        # Post-process to clean up SPARQL output (mimicking sed commands in Makefile)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .str.replace("?", "", regex=False)
                    .str.replace("<http://purl.obolibrary.org/obo/MONDO_", "MONDO:", regex=False)
                    .str.replace("http://purl.obolibrary.org/obo/mondo#", "mondo:", regex=False)
                    .str.replace(">", "", regex=False)
                )
        df.columns = [col.replace("?", "") for col in df.columns]
        logger.info(f"Extracted metrics for {len(df)} diseases")
        return df

def matrix_disease_filter(df_disease_list_unfiltered):
    """
    Filter a DataFrame by a specific column and value.
    
    Parameters
    ----------
    df_disease_list_unfiltered : pandas.DataFrame
        The disease list, unfiltered, but with columns that are used for filtering.
    
    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        A tuple of two DataFrames: the first one contains the included diseases, and the second one contains the excluded diseases.
    """
    filter_column = 'official_matrix_filter'
    
    # By default, no disease is included
    df_disease_list_unfiltered[filter_column] = False
    
    # QC: Check for conflicts where both f_matrix_manually_included and f_matrix_manually_excluded are True
    conflicts = df_disease_list_unfiltered[
        df_disease_list_unfiltered['f_matrix_manually_included'] & df_disease_list_unfiltered['f_matrix_manually_excluded']
    ]

    if not conflicts.empty:
        # Format the conflicts nicely
        conflict_str = conflicts.to_string(index=False)
        raise ValueError(f"Conflicts found: The following entries are marked as both manually included and manually excluded:\n{conflict_str}")
    
    # First, we add all manually curated classes to the list
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered['f_matrix_manually_included'] == True
    
    # Next, we add all leaf classes
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered['f_leaf'] == True

    # Now, we add all the immediate parents of leaf classes that are mapped to OMIM, ICD, or Orphanet
    df_disease_list_unfiltered[filter_column] |= (
        (df_disease_list_unfiltered['f_leaf_direct_parent'] == True) & 
        (
            (df_disease_list_unfiltered['f_omim'] == True) | 
            (df_disease_list_unfiltered['f_omimps_descendant'] == True) | 
            (df_disease_list_unfiltered['f_icd_category'] == True) |
            (df_disease_list_unfiltered['f_orphanet_disorder'] == True) |
            (df_disease_list_unfiltered['f_orphanet_subtype'] == True)
        )
    )
    
    # Next, we add all diseases corresponding to ICD 10 billable codes
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered['f_icd_category'] == True
    
    # Next, we add all diseases corresponding to Orphanet disorders
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered['f_orphanet_disorder'] == True
    
    # Next, we add all diseases corresponding to ClinGen curated conditions
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered['f_clingen'] == True
    
    # Next, we add all diseases corresponding to OMIM curated diseases
    df_disease_list_unfiltered[filter_column] |= df_disease_list_unfiltered['f_omim'] == True
    
    # Next we remove all susceptibilities, mondo subtypes, and diseases with and/or or with/without
    # UPDATE 13.02.2025: We will for now _not_ remove these, but provide filter columns for them instead
    #df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_mondo_subtype'] == True, filter_column] = False
    #df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_susceptibility'] == True, filter_column] = False
    #df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_andor'] == True, filter_column] = False
    #df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_withorwithout'] == True, filter_column] = False
    
    ## Remove all hereditary diseases without classification. This is imo a dangerous default, but
    ## @jxneli reviewed all 849 cases from the February 2025 release and found that all were indeed
    ## "irrelevant" for drug repurposing, https://github.com/everycure-org/matrix-disease-list/issues/50
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_unclassified_hereditary'] == True, filter_column] = False
    
    ## Remove all diseases that are candidates for obsoletion
    ## https://github.com/everycure-org/matrix-disease-list/issues/48
    # df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_obsoletion_candidate'] == True, filter_column] = False
    
    ## Remove all paraphilic disorders
    ## https://github.com/everycure-org/matrix-disease-list/issues/42
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_paraphilic'] == True, filter_column] = False
    
    
    # Remove disease that were manually excluded
    df_disease_list_unfiltered.loc[df_disease_list_unfiltered['f_matrix_manually_excluded'] == True, filter_column] = False
    
    return df_disease_list_unfiltered 


def matrix_filter_final_columns(df_disease_list):
    """
    Filter a DataFrame by a specific column and value.
    
    Parameters
    ----------
    df_disease_list : pandas.DataFrame
        The disease list with columns that are used for filtering.
    
    Returns
    -------
    pandas.DataFrame
        The disease list with only the relevant columns.
    """
    return df_disease_list[['category_class', 'label', 'definition', 'synonyms', 'subsets', 'crossreferences']]


def extract_groupings(subsets, groupings):
    """Extract groupings for list of subsets."""
    result = {grouping: [] for grouping in groupings}
    
    if subsets:
        for subset in subsets.split(";"):
            subset = subset.strip()
            for grouping in groupings:
                if subset.startswith(f"mondo:{grouping}"):
                    subset_tag = subset.replace("mondo:","").replace(grouping,"").replace(" ","").strip("_")
                    if (subset_tag != "member") and (subset_tag != ""):
                        result[grouping].append(subset_tag.replace("|",""))
    
    # This looks very complex: if there are multiple values, we join them with a pipe, but we exclude "other" in this case
    return {
        key: "|".join([v for v in values if v != "other"] if len(values) > 1 else values) if values else ""
        for key, values in result.items()
    }
    
def is_grouping_heuristic(df):
    col_out = "is_grouping_heuristic"

    not_grouping = [
        "is_clingen",
        "is_orphanet_subtype",
        "is_orphanet_subtype_descendant",
        "is_omimps_descendant",
        "is_leaf",
        "is_leaf_direct_parent",
        "is_orphanet_disorder",
        "is_omim",
        "is_icd_billable",
        "is_mondo_subtype",
    ]

    grouping = [
        "is_grouping_subset",
        "is_grouping_subset_ancestor",
        "is_omimps",
        "is_icd_chapter_header",
        "is_icd_chapter_code",
    ]
    
    # Ensure all expected grouping-related columns are present
    expected_cols = set(grouping) | set(not_grouping)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logging.warning("{df.columns}")
        raise ValueError(f"DataFrame is missing expected grouping columns: {', '.join(sorted(missing))}")

    # Initialize the column to False
    df[col_out] = False

    # Set True if any grouping column is True
    for col in grouping:
        if col in df.columns:
            df[col_out] = df[col_out] | df[col].fillna(False)

    # Override by setting False if any not_grouping column is True
    for col in not_grouping:
        if col in df.columns:
            df.loc[df[col].fillna(False), col_out] = False

    return df


def create_disease_list(
    disease_list_raw: pd.DataFrame,
    mondo_metrics: pd.DataFrame,
    subtype_counts: str,
) -> Dict[str, pd.DataFrame]:
    """Apply filters to create final disease lists.
    """
    logger.info("Create final disease list")

    # Load the TSV file
    subtype_group_counts = subtype_counts[["subset_group_id", "other_subsets_count"]].drop_duplicates()
    subtype_group_counts.columns = ["category_class", "count_subtypes"]
    
    # Filter the DataFrame
    df_matrix_disease_filter_modified = matrix_disease_filter(disease_list_raw)
            
    curated_disease_groupings = ["harrisons_view", "mondo_txgnn", "mondo_top_grouping"]
    llm_disease_groupings = [ "medical_specialization",	"txgnn", "anatomical", "is_pathogen_caused", "is_cancer", "is_glucose_dysfunction", "tag_existing_treatment", "tag_qaly_lost"]
    disease_groupings = curated_disease_groupings + llm_disease_groupings
    df_disease_groupings = df_matrix_disease_filter_modified[["category_class", "label", "subsets"]]
    
    # Apply the function to extract groupings
    df_disease_groupings_extracted = df_disease_groupings["subsets"].apply(
        lambda x: extract_groupings(x, disease_groupings)
    ).apply(pd.Series)

    # Combine with the original DataFrame
    df_disease_groupings_pivot = pd.concat(
        [df_disease_groupings[["category_class", "label"]], df_disease_groupings_extracted], axis=1
    )
    df_disease_groupings_pivot.sort_values(by="category_class", inplace=True)
    df_disease_groupings_pivot.drop(columns=['label'], inplace=True)
       
    # As per convention, rewrite filter columns to is_ 
    # https://github.com/everycure-org/matrix-disease-list/issues/75
    df_matrix_disease_filter_modified.rename(columns=lambda x: re.sub(r'^f_', 'is_', x) if x.startswith("f_") else x, inplace=True)
    
    df_matrix_disease_filter_modified = df_matrix_disease_filter_modified.merge(df_disease_groupings_pivot, on='category_class', how='left')
    df_matrix_disease_filter_modified = df_matrix_disease_filter_modified.merge(mondo_metrics, on='category_class', how='left')
    df_matrix_disease_filter_modified = df_matrix_disease_filter_modified.merge(subtype_group_counts, on='category_class', how='left')
    df_matrix_disease_filter_modified = df_matrix_disease_filter_modified.merge(subtype_counts[["subset_id", "subset_group_id", "subset_group_label", "other_subsets_count"]], left_on='category_class', right_on="subset_id", how='left')
    df_matrix_disease_filter_modified['count_subtypes'] = df_matrix_disease_filter_modified['count_subtypes'].fillna(0)
    df_matrix_disease_filter_modified['count_descendants'] = df_matrix_disease_filter_modified['count_descendants'].fillna(0)
    df_matrix_disease_filter_modified['count_descendants_without_subtypes'] = (
        df_matrix_disease_filter_modified['count_descendants'] - df_matrix_disease_filter_modified['count_subtypes']
    )
    
    # Remove subset_id column after merge
    df_matrix_disease_filter_modified.drop(columns=['subset_id'], inplace=True)

    # Given all columns with start with is_ or tag_ should be boolean, we convert them to True/False
    columns_to_check = [col for col in df_matrix_disease_filter_modified.columns if col.startswith('is_')]
    
    # Model this exceptoon, hopefully it will go away in a future iteration:
    # https://github.com/everycure-org/matrix-disease-list/issues/75
    
    if "tag_existing_treatment" in df_matrix_disease_filter_modified.columns:
        columns_to_check.append('tag_existing_treatment')
    
    # Compute a general grouping heuristic for the all diseases in the list
    df_matrix_disease_filter_modified = is_grouping_heuristic(df_matrix_disease_filter_modified)
    
    for col in columns_to_check:
        df_matrix_disease_filter_modified[col] = df_matrix_disease_filter_modified[col].map(
            lambda x: 'True' if isinstance(x, bool) and x or (isinstance(x, str) and x.lower() == 'true') else 'False')
    
    return {
        "disease_list": df_matrix_disease_filter_modified,
    }


def extract_mondo_metadata(
    mondo_owl: str,
    sparql_query_path: str,
) -> pd.DataFrame:
    """Extract metadata about MONDO version using ROBOT query.

    Args:
        mondo_owl: MONDO ontology content
        sparql_query_path: Path to SPARQL query file (ontology-metadata.sparql)

    Returns:
        DataFrame with MONDO metadata (version, IRI, etc.)
    """
    logger.info("Extracting MONDO metadata")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file
        input_owl = tmpdir_path / "mondo.owl"
        output_tsv = tmpdir_path / "metadata.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_owl)

        # Run ROBOT query command
        cmd = f'robot query -i "{input_owl}" -f tsv --query "{sparql_query_path}" "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output TSV with dtype specification to handle mixed types
        df = pd.read_csv(output_tsv, sep="\t", dtype=str, na_filter=False)

        # Post-process to clean up SPARQL output
        for col in df.columns:
            df[col] = (
                df[col]
                .str.replace("?", "", regex=False)
                .str.replace("<", "", regex=False)
                .str.replace(">", "", regex=False)
            )

        logger.info("Extracted MONDO metadata")
        return df


def extract_mondo_obsoletes(
    mondo_owl: str,
    sparql_query_path: str,
) -> pd.DataFrame:
    """Extract list of obsolete MONDO terms using ROBOT query.

    Args:
        mondo_owl: MONDO ontology content
        sparql_query_path: Path to SPARQL query file (mondo-obsoletes.sparql)

    Returns:
        DataFrame with obsolete disease IDs
    """
    logger.info("Extracting MONDO obsoletes")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file 
        # TODO is this necessary or can we pass the path directly?
        input_owl = tmpdir_path / "mondo.owl"
        output_tsv = tmpdir_path / "obsoletes.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_owl)

        # Run ROBOT query command
        cmd = f'robot query -i "{input_owl}" -f tsv --query "{sparql_query_path}" "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output TSV with dtype specification to handle mixed types
        df = pd.read_csv(output_tsv, sep="\t", dtype=str, na_filter=False)

        # Post-process to clean up SPARQL output (mimicking sed commands from Makefile lines 128-132)
        for col in df.columns:
            df[col] = (
                df[col]
                .str.replace("?", "", regex=False)
                .str.replace("<", "", regex=False)
                .str.replace(">", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.replace("http://purl.obolibrary.org/obo/MONDO_", "MONDO:", regex=False)
            )

        logger.info(f"Extracted {len(df)} obsolete MONDO terms")
        return df


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
        raise ValueError(f"Validation failed: Disease list has only {len(disease_list)} entries, expected at least 15000")
    
    # check for duplicate MONDO IDs
    if disease_list['category_class'].duplicated().any():
        duplicated_ids = disease_list[disease_list['category_class'].duplicated()]['category_class'].unique()
        raise ValueError(f"Validation failed: Duplicate MONDO IDs found: {duplicated_ids}")


