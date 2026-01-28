"""Disease list pipeline for MATRIX.

This pipeline creates the MATRIX disease list from the MONDO disease ontology.
It processes raw ontology data, applies filters, and generates the final
curated list of diseases for drug repurposing.
"""

from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs: Any) -> Pipeline:
    """Create the disease list generation pipeline.

    The pipeline has three main stages:
    1. Data preparation: Extract labels, create templates
    2. Ontology processing: Merge templates into MONDO
    3. List generation: Apply filters and create final outputs

    Args:
        **kwargs: Additional keyword arguments (unused, for Kedro compatibility)

    Returns:
        Complete disease list pipeline
    """
    return pipeline(
        [
            # Stage 1: Ingest data outside of Mondo
            node(
                func=nodes.ingest_obsoletion_candidates,
                inputs={
                    "mondo_obsoletion_candidates": "disease_mondo.raw.mondo_obsoletion_candidates",
                },
                outputs="disease_mondo.prm.mondo_obsoletion_candidates",
                name="ingest_obsoletion_candidates",
                tags=["disease_mondo", "preparation"],
            ),
            node(
                func=nodes.create_billable_icd10_codes,
                inputs={
                    "icd10_codes": "disease_mondo.raw.icd10_cm_codes",
                    "mondo_sssom": "disease_mondo.raw.mondo_sssom",
                    "icd10_billable_subset": "params:disease_mondo.icd10_billable_subset",
                    "icd10cm_prefix": "params:disease_mondo.icd10cm_prefix",
                },
                outputs="disease_mondo.int.billable_icd10_codes",
                name="create_billable_icd10_codes",
                tags=["disease_mondo", "preparation"],
            ),
            # Stage 2: Extract all disease data from MONDO
            node(
                func=nodes.extract_disease_data_from_mondo,
                inputs={
                    "mondo_graph": "disease_mondo.raw.mondo_graph",
                    "billable_icd10": "disease_mondo.int.billable_icd10_codes",
                    "subtypes_params": "params:disease_mondo.subtypes_params",
                    "subtype_patterns": "params:disease_mondo.subtype_patterns",
                },
                outputs={
                    "mondo_metadata": "disease_mondo.prm.mondo_metadata",
                    "mondo_obsoletes": "disease_mondo.prm.mondo_obsoletes",
                    "disease_list_raw": "disease_mondo.int.disease_list_raw",
                    "mondo_metrics": "disease_mondo.int.mondo_metrics",
                    "subtype_counts": "disease_mondo.int.subtype_counts",
                },
                name="extract_disease_data_from_mondo",
                tags=["disease_mondo", "extraction"],
            ),
            # Stage 3: Compute the final disease list
            node(
                func=nodes.create_mondo_disease_list,
                inputs={
                    "disease_list_raw": "disease_mondo.int.disease_list_raw",
                    "mondo_metrics": "disease_mondo.int.mondo_metrics",
                    "subtype_counts": "disease_mondo.int.subtype_counts",
                    "parameters": "params:disease_mondo.disease_list_params",
                },
                outputs={
                    "disease_list": "disease_mondo.prm.disease_list",
                },
                name="create_mondo_disease_list",
                tags=["disease_mondo"],
            ),
        ]
    )
