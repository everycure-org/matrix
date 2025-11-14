"""Disease list pipeline for MATRIX.

This pipeline creates the MATRIX disease list from the MONDO disease ontology.
It processes raw ontology data, applies filters, and generates the final
curated list of diseases for drug repurposing.
"""

from pathlib import Path
from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

# Get path to queries directory
QUERIES_DIR = Path(__file__).parent / "queries"


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
            # Stage 1: Extract billable ICD-10 codes and map to Mondo
            node(
                func=nodes.create_billable_icd10_codes,
                inputs={
                    "icd10_codes": "disease_list.raw.icd10_cm_codes",
                    "mondo_sssom": "disease_list.raw.mondo_sssom",
                    "parameters": "params:billable_icd10_params",
                },
                outputs="disease_list.int.billable_icd10_codes",
                name="create_billable_icd10_codes",
                tags=["disease_list", "preparation"],
            ),
            # Stage 2: Extract all disease data from MONDO
            node(
                func=nodes.extract_disease_data_from_mondo,
                inputs={
                    "mondo_graph": "disease_list.raw.mondo_graph",
                    "billable_icd10": "disease_list.int.billable_icd10_codes",
                    "subtypes_params": "params:subtypes_params",
                    "subtype_patterns": "params:subtype_patterns",
                },
                outputs={
                    "mondo_metadata": "disease_list.prm.mondo_metadata",
                    "mondo_obsoletes": "disease_list.prm.mondo_obsoletes",
                    "disease_list_raw": "disease_list.int.disease_list_raw",
                    "mondo_metrics": "disease_list.int.mondo_metrics",
                    "mondo_preprocessed": "disease_list.int.mondo_preprocessed",
                    "subtype_counts": "disease_list.int.subtype_counts",
                },
                name="extract_disease_data_from_mondo",
                tags=["disease_list", "extraction"],
            ),
            # Stage 3: Compute the final disease list
            node(
                func=nodes.create_disease_list,
                inputs={
                    "disease_list_raw": "disease_list.int.disease_list_raw",
                    "mondo_metrics": "disease_list.int.mondo_metrics",
                    "subtype_counts": "disease_list.int.subtype_counts",
                    "parameters": "params:disease_list_params",
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
