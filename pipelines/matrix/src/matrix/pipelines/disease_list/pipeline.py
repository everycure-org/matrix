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
            node(
                func=nodes.extract_mondo_labels,
                inputs="disease_list.raw.mondo_owl",
                outputs="disease_list.int.mondo_labels",
                name="extract_mondo_labels",
                tags=["disease_list", "preparation"],
            ),
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
            node(
                func=nodes.merge_templates_into_mondo,
                inputs={
                    "mondo_owl": "disease_list.raw.mondo_owl",
                    "included_diseases": "disease_list.raw.included_diseases_template",
                    "excluded_diseases": "disease_list.raw.excluded_diseases_template",
                    "grouping_diseases": "disease_list.raw.grouping_diseases_template",
                    "billable_icd10": "disease_list.int.billable_icd10_template",
                    "subtypes": "disease_list.int.subtypes_template",
                },
                outputs="disease_list.int.mondo_with_subsets",
                name="merge_templates_into_mondo",
                tags=["disease_list", "ontology_processing"],
            ),
            node(
                func=nodes.extract_disease_list_unfiltered,
                inputs={
                    "mondo_with_subsets": "disease_list.int.mondo_with_subsets",
                    "sparql_query_path": str(QUERIES_DIR / "matrix-disease-list-filters.sparql"),
                },
                outputs="disease_list.int.disease_list_unfiltered",
                name="extract_disease_list_unfiltered",
                tags=["disease_list", "ontology_processing"],
            ),
            node(
                func=nodes.extract_disease_metrics,
                inputs={
                    "mondo_with_subsets": "disease_list.int.mondo_with_subsets",
                    "sparql_query_path": str(QUERIES_DIR / "matrix-disease-list-metrics.sparql"),
                },
                outputs="disease_list.int.disease_list_metrics",
                name="extract_disease_metrics",
                tags=["disease_list", "ontology_processing"],
            ),
            # Stage 3: List generation
            node(
                func=nodes.apply_disease_filters,
                inputs={
                    "disease_list_unfiltered": "disease_list.int.disease_list_unfiltered",
                    "disease_metrics": "disease_list.int.disease_list_metrics",
                },
                outputs={
                    "disease_list": "disease_list.prm.disease_list",
                    "excluded_diseases": "disease_list.prm.excluded_diseases",
                    "disease_groupings": "disease_list.prm.disease_groupings",
                },
                name="apply_disease_filters",
                tags=["disease_list", "filtering"],
            ),
            # Metadata extraction
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
            # Validation
            node(
                func=nodes.validate_disease_list,
                inputs={
                    "disease_list": "disease_list.prm.disease_list",
                    "excluded_diseases": "disease_list.prm.excluded_diseases",
                },
                outputs=None,
                name="validate_disease_list",
                tags=["disease_list", "validation"],
            ),
            # Reporting
            node(
                func=nodes.generate_disease_list_report,
                inputs={
                    "disease_list": "disease_list.prm.disease_list",
                    "excluded_diseases": "disease_list.prm.excluded_diseases",
                    "disease_groupings": "disease_list.prm.disease_groupings",
                    "mondo_metadata": "disease_list.prm.mondo_metadata",
                },
                outputs=None,
                name="generate_disease_list_report",
                tags=["disease_list", "reporting"],
            ),
        ]
    )
