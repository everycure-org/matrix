"""Disease list pipeline for MATRIX.

This pipeline creates the MATRIX disease list from the MONDO disease ontology.
It processes raw ontology data, applies filters, and generates the final
curated list of diseases for drug repurposing.
"""

from pathlib import Path

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

# Get path to queries directory
QUERIES_DIR = Path(__file__).parent / "queries"


def create_pipeline(**kwargs) -> Pipeline:
    """Create the disease list generation pipeline.

    The pipeline has three main stages:
    1. Data preparation: Extract labels, create templates
    2. Ontology processing: Merge templates into MONDO
    3. List generation: Apply filters and create final outputs

    Returns:
        Complete disease list pipeline
    """
    return pipeline(
        [
            # Stage 1: Data preparation
            # Stage 1: Data preparation
            # Extract labels from MONDO ontology for disease identification
            node(
                func=nodes.extract_mondo_labels,
                inputs="disease_list.raw.mondo_owl",
                outputs="disease_list.int.mondo_labels",
                name="extract_mondo_labels",
                tags=["disease_list", "preparation"],
            ),
            # Create template for billable ICD-10 codes by mapping MONDO to ICD-10-CM
            node(
                func=nodes.create_billable_icd10_template,
                inputs={
                    "icd10_codes": "disease_list.raw.icd10_cm_codes",
                    "mondo_sssom": "disease_list.raw.mondo_sssom",
                },
                outputs="disease_list.int.billable_icd10_template",
                name="create_billable_icd10_template",
                tags=["disease_list", "preparation"],
            ),
            # Create template for disease subtypes based on MONDO hierarchy
            node(
                func=nodes.create_subtypes_template,
                inputs={
                    "mondo_labels": "disease_list.int.mondo_labels",
                    "mondo_owl": "disease_list.raw.mondo_owl",
                },
                outputs="disease_list.int.subtypes_template",
                name="create_subtypes_template",
                tags=["disease_list", "preparation"],
            ),
            # Stage 2: Ontology processing
            # Merge ICD-10 and subtype templates into MONDO ontology
            node(
                func=nodes.merge_templates_into_mondo,
                inputs={
                    "mondo_owl": "disease_list.raw.mondo_owl",
                    "billable_icd10": "disease_list.int.billable_icd10_template",
                    "subtypes": "disease_list.int.subtypes_template",
                },
                outputs="disease_list.int.mondo_preprocessed",
                name="merge_templates_into_mondo",
                tags=["disease_list", "ontology_processing"],
            ),
            # Extract unfiltered disease list using SPARQL query
            node(
                func=nodes.extract_disease_list_raw,
                inputs=["disease_list.int.mondo_preprocessed", "params:queries.matrix_disease_list_filters"],
                outputs="disease_list.int.disease_list_raw",
                name="extract_disease_list_raw",
                tags=["disease_list", "ontology_processing"],
            ),
            # Extract disease metrics for filtering decisions
            node(
                func=nodes.extract_mondo_metrics,
                inputs=["disease_list.int.mondo_preprocessed", "params:queries.matrix_disease_list_metrics"],
                outputs="disease_list.int.mondo_metrics",
                name="extract_mondo_metrics",
                tags=["disease_list", "ontology_processing"],
            ),
            # Stage 3: List generation
            # Apply filtering rules to generate final disease list
            node(
                func=nodes.apply_disease_filters,
                inputs={
                    "disease_list_unfiltered": "disease_list.int.disease_list_raw",
                    "mondo_metrics": "disease_list.int.mondo_metrics",
                },
                outputs={
                    "disease_list": "disease_list.prm.disease_list",
                    "excluded_diseases": "disease_list.prm.excluded_diseases",
                    "disease_groupings": "disease_list.prm.disease_groupings",
                },
                name="apply_disease_filters",
                tags=["disease_list", "filtering"],
            ),
            # Stage 4: Metadata extraction
            # Extract ontology metadata (version, date, etc.)
            node(
                func=nodes.extract_mondo_metadata,
                inputs={
                    "mondo_owl": "disease_list.raw.mondo_owl",
                    "sparql_query_path": str(QUERIES_DIR / "ontology-metadata.sparql"),
                },
                outputs="disease_list.prm.mondo_metadata",
                name="extract_mondo_metadata",
                tags=["disease_list", "metadata"],
            ),
            # Extract obsolete MONDO terms for tracking deprecated diseases
            node(
                func=nodes.extract_mondo_obsoletes,
                inputs={
                    "mondo_owl": "disease_list.raw.mondo_owl",
                    "sparql_query_path": str(QUERIES_DIR / "mondo-obsoletes.sparql"),
                },
                outputs="disease_list.prm.mondo_obsoletes",
                name="extract_mondo_obsoletes",
                tags=["disease_list", "metadata"],
            ),
            # Stage 4: Validation
            node(
                func=nodes.validate_disease_list,
                inputs={
                    "disease_list": "disease_list.prm.disease_list",
                },
                outputs=None,
                name="validate_disease_list",
                tags=["disease_list", "validation"],
            ),
        ]
    )
