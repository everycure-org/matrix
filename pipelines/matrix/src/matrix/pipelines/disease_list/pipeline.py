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
                inputs={
                    "mondo_owl": "disease_list.raw.mondo_owl",
                    "ontology_metadata_query": "params:queries.ontology_metadata",
                    "mondo_obsoletes_query": "params:queries.mondo_obsoletes",
                },
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
                    "exact_match_predicate": "params:robot_template.exact_match_predicate",
                    "icd10_billable_subset": "params:robot_template.icd10_billable_subset",
                    "subset_annotation": "params:robot_template.subset_annotation",
                    "icd10cm_prefix": "params:robot_template.icd10cm_prefix",
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
                    "chromosomal_diseases_root": "params:mondo_ids.chromosomal_diseases_root",
                    "human_diseases_root": "params:mondo_ids.human_diseases_root",
                    "chromosomal_diseases_exceptions": "params:mondo_ids.chromosomal_diseases_exceptions",
                    "subtype_patterns": "params:subtype_patterns",
                    "mondo_prefix": "params:robot_template.mondo_prefix",
                    "mondo_subtype_subset": "params:robot_template.mondo_subtype_subset",
                    "default_contributor": "params:robot_template.default_contributor",
                    "subset_annotation_split": "params:robot_template.subset_annotation_split",
                    "contributor_annotation": "params:robot_template.contributor_annotation",
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
                    "inject_mondo_top_grouping_query": "params:queries.inject_mondo_top_grouping",
                    "inject_susceptibility_subset_query": "params:queries.inject_susceptibility_subset",
                    "inject_subset_declaration_query": "params:queries.inject_subset_declaration",
                    "downfill_disease_groupings_query": "params:queries.downfill_disease_groupings",
                    "disease_groupings_other_query": "params:queries.disease_groupings_other",
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
