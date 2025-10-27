"""Disease list pipeline nodes for MATRIX.

This pipeline processes the MONDO disease ontology to create the MATRIX disease list,
which defines which diseases are included in the drug repurposing platform.

The workflow is modeled after the matrix-disease-list repository:
https://github.com/everycure-org/matrix-disease-list
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from matrix.utils.system import run_subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_mondo_labels(mondo_owl: str) -> pd.DataFrame:
    """Extract disease labels from MONDO ontology using ROBOT.

    Args:
        mondo_owl: Path to MONDO ontology in OWL format

    Returns:
        DataFrame with disease IDs and labels
    """
    logger.info("Extracting MONDO labels from ontology")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # Run ROBOT export command to extract ID and LABEL columns
        cmd = f'robot export -i "{mondo_owl}" -f tsv --header "ID|LABEL" --export "{output_path}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the exported TSV
        df = pd.read_csv(output_path, sep="\t")
        logger.info(f"Extracted {len(df)} MONDO labels")

        return df
    finally:
        # Clean up temporary file
        Path(output_path).unlink(missing_ok=True)


def create_billable_icd10_template(
    icd10_codes: str,
    mondo_sssom: pd.DataFrame,
) -> pd.DataFrame:
    """Create template for billable ICD-10-CM codes mapped to MONDO.

    Args:
        icd10_codes: ICD-10-CM codes file
        mondo_sssom: MONDO SSSOM mappings

    Returns:
        DataFrame with billable ICD-10 codes template
    """
    logger.info("Creating billable ICD-10-CM template")
    # TODO: Implement ICD-10-CM code extraction and mapping to MONDO
    # Should match logic from scripts/matrix-disease-list.py create-billable-icd10-template
    raise NotImplementedError("create_billable_icd10_template not yet implemented")


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
    # TODO: Implement subtype extraction logic
    # Should identify diseases with high granularity subtypes
    # Match logic from create-template-with-high-granularity-subtypes
    raise NotImplementedError("create_subtypes_template not yet implemented")


def merge_templates_into_mondo(
    mondo_owl: str,
    included_diseases: pd.DataFrame,
    excluded_diseases: pd.DataFrame,
    grouping_diseases: pd.DataFrame,
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
        included_diseases: Manually curated included diseases template
        excluded_diseases: Manually curated excluded diseases template
        grouping_diseases: Disease grouping annotations template
        billable_icd10: Billable ICD-10 template
        subtypes: Subtypes template

    Returns:
        Path to modified MONDO ontology with subset annotations
    """
    logger.info("Merging disease templates into MONDO ontology")

    # Create temporary files for templates
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write DataFrames to temporary template files
        included_path = tmpdir_path / "included.robot.tsv"
        excluded_path = tmpdir_path / "excluded.robot.tsv"
        grouping_path = tmpdir_path / "grouping.robot.tsv"
        billable_path = tmpdir_path / "billable.robot.tsv"
        subtypes_path = tmpdir_path / "subtypes.robot.tsv"
        output_path = tmpdir_path / "mondo-with-subsets.owl"

        included_diseases.to_csv(included_path, sep="\t", index=False)
        excluded_diseases.to_csv(excluded_path, sep="\t", index=False)
        grouping_diseases.to_csv(grouping_path, sep="\t", index=False)
        billable_icd10.to_csv(billable_path, sep="\t", index=False)
        subtypes.to_csv(subtypes_path, sep="\t", index=False)

        # Build ROBOT command to merge all templates
        # Note: We'll start simple - just template merging without SPARQL updates for now
        # SPARQL queries will need to be added based on the actual query files
        cmd = f"""robot template -i "{mondo_owl}" --merge-after \
            --template "{excluded_path}" \
            --template "{included_path}" \
            --template "{grouping_path}" \
            --template "{billable_path}" \
            --template "{subtypes_path}" \
            -o "{output_path}" """

        logger.info("Running ROBOT template merge command")
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output file and return its contents as a string
        # For now, we'll return the path - the catalog should handle this as text.TextDataset
        with open(output_path, "r") as f:
            owl_content = f.read()

        logger.info("Successfully merged templates into MONDO ontology")
        return owl_content


def extract_disease_list_unfiltered(
    mondo_with_subsets: str,
    sparql_query_path: str,
) -> pd.DataFrame:
    """Extract unfiltered disease list with filter features using ROBOT query.

    Args:
        mondo_with_subsets: MONDO ontology content with subset annotations
        sparql_query_path: Path to SPARQL query file (matrix-disease-list-filters.sparql)

    Returns:
        DataFrame with unfiltered disease list
    """
    logger.info("Extracting unfiltered disease list")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file
        input_owl = tmpdir_path / "mondo-with-subsets.owl"
        output_tsv = tmpdir_path / "disease-list-unfiltered.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_with_subsets)

        # Run ROBOT query command
        cmd = f'robot query -i "{input_owl}" -f tsv --query "{sparql_query_path}" "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output TSV
        df = pd.read_csv(output_tsv, sep="\t")

        # Post-process to clean up SPARQL output (mimicking sed commands in Makefile)
        # Remove ? prefix, angle brackets, and clean up IRIs
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .str.replace("?", "", regex=False)
                    .str.replace("<http://purl.obolibrary.org/obo/MONDO_", "MONDO:", regex=False)
                    .str.replace("http://purl.obolibrary.org/obo/mondo#", "mondo:", regex=False)
                    .str.replace(">", "", regex=False)
                )

        logger.info(f"Extracted {len(df)} diseases in unfiltered list")
        return df


def extract_disease_metrics(
    mondo_with_subsets: str,
    sparql_query_path: str,
) -> pd.DataFrame:
    """Extract metrics for disease list filtering using ROBOT query.

    Args:
        mondo_with_subsets: MONDO ontology content with subset annotations
        sparql_query_path: Path to SPARQL query file (matrix-disease-list-metrics.sparql)

    Returns:
        DataFrame with disease metrics
    """
    logger.info("Extracting disease metrics")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write OWL content to temporary file
        input_owl = tmpdir_path / "mondo-with-subsets.owl"
        output_tsv = tmpdir_path / "disease-metrics.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_with_subsets)

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

        logger.info(f"Extracted metrics for {len(df)} diseases")
        return df


def apply_disease_filters(
    disease_list_unfiltered: pd.DataFrame,
    disease_metrics: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Apply filters to create final disease lists.

    Args:
        disease_list_unfiltered: Unfiltered disease list with features
        disease_metrics: Disease metrics for filtering

    Returns:
        Dictionary containing:
        - disease_list: Final filtered disease list
        - excluded_diseases: Diseases excluded from list
        - disease_groupings: Disease grouping information
    """
    logger.info("Applying disease filters")
    # TODO: Implement filtering logic
    # Should match scripts/matrix-disease-list.py create-matrix-disease-list
    # Returns multiple outputs: included, excluded, groupings
    raise NotImplementedError("apply_disease_filters not yet implemented")


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

        # Read the output TSV
        df = pd.read_csv(output_tsv, sep="\t")

        # Post-process to clean up SPARQL output
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace("?", "", regex=False).str.replace("<", "", regex=False).str.replace(">", "", regex=False)

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
        input_owl = tmpdir_path / "mondo.owl"
        output_tsv = tmpdir_path / "obsoletes.tsv"

        with open(input_owl, "w") as f:
            f.write(mondo_owl)

        # Run ROBOT query command
        cmd = f'robot query -i "{input_owl}" -f tsv --query "{sparql_query_path}" "{output_tsv}"'
        run_subprocess(cmd, check=True, stream_output=True)

        # Read the output TSV
        df = pd.read_csv(output_tsv, sep="\t")

        # Post-process to clean up SPARQL output (mimicking sed commands from Makefile lines 128-132)
        for col in df.columns:
            if df[col].dtype == object:
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
    excluded_diseases: pd.DataFrame,
) -> bool:
    """Validate the generated disease list.

    Args:
        disease_list: Final disease list
        excluded_diseases: Excluded diseases list

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating disease list")
    # TODO: Implement validation checks
    # - No duplicates
    # - No overlap between included and excluded
    # - All required columns present
    # - Valid MONDO IDs
    raise NotImplementedError("validate_disease_list not yet implemented")


def generate_disease_list_report(
    disease_list: pd.DataFrame,
    excluded_diseases: pd.DataFrame,
    disease_groupings: pd.DataFrame,
    mondo_metadata: pd.DataFrame,
) -> str:
    """Generate summary report about the disease list.

    Args:
        disease_list: Final disease list
        excluded_diseases: Excluded diseases
        disease_groupings: Disease groupings
        mondo_metadata: MONDO version metadata

    Returns:
        Markdown formatted report
    """
    logger.info("Generating disease list report")
    # TODO: Implement report generation
    # Should include:
    # - Number of included/excluded diseases
    # - Breakdown by disease grouping
    # - MONDO version information
    # - Comparison with previous version if available
    raise NotImplementedError("generate_disease_list_report not yet implemented")
