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
            # Extract labels, metadata, and obsoletes from Mondo ontology
            node(
                func=nodes.extract_metadata_from_mondo,
                inputs="disease_list.raw.mondo_owl",
                outputs={
                    "mondo_labels": "disease_list.int.mondo_labels",
                    "mondo_metadata": "disease_list.prm.mondo_metadata",
                    "mondo_obsoletes": "disease_list.prm.mondo_obsoletes",
                },
                name="extract_metadata_from_mondo",
                tags=["disease_list", "preparation"],
            ),
            # Prepare Mondo metadata for billable ICD-10 codes by mapping MONDO to ICD-10-CM
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
            # Prepare Mondo metadata for disease subtypes based on MONDO hierarchy
            # TODO uses OAK not ROBOT, so maybe can stream?
            node(
                func=nodes.create_subtypes_template,
                inputs={
                    "mondo_labels": "disease_list.int.mondo_labels",
                    "mondo_owl": "disease_list.raw.mondo_owl",
                },
                outputs={
                    "subtypes_template": "disease_list.int.subtypes_template",
                    "subtypes_counts": "disease_list.int.subtypes_counts",
                },
                name="create_subtypes_template",
                tags=["disease_list", "preparation"],
            ),
            # Stage 2: Extract updated information from preprocessed Mondo ontology
            node(
                func=nodes.process_mondo_with_templates,
                inputs={
                    "mondo_owl": "disease_list.raw.mondo_owl",
                    "billable_icd10": "disease_list.int.billable_icd10_template",
                    "subtypes": "disease_list.int.subtypes_template",
                    "disease_list_filters_query": "params:queries.matrix_disease_list_filters",
                    "metrics_query": "params:queries.matrix_disease_list_metrics",
                },
                outputs={
                    "disease_list_raw": "disease_list.int.disease_list_raw",
                    "mondo_metrics": "disease_list.int.mondo_metrics",
                    "mondo_preprocessed": "disease_list.int.mondo_preprocessed",
                },
                name="process_mondo_with_templates",
                tags=["disease_list", "ontology_processing"],
            ),
            # Stage 3: Compute the final disease list
            node(
                func=nodes.create_disease_list,
                inputs={
                    "disease_list_raw": "disease_list.int.disease_list_raw",
                    "mondo_metrics": "disease_list.int.mondo_metrics",
                    "subtype_counts": "disease_list.int.subtypes_counts",
                },
                outputs={
                    "disease_list": "disease_list.prm.disease_list",
                },
                name="create_disease_list",
                tags=["disease_list"],
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
