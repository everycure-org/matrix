"""This module creates a pipeline for the BTE process."""

from kedro.pipeline import Pipeline, node
from .nodes import run_bte_queries


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for the BTE process.

    Args:
        kwargs: Additional keyword arguments.

    Returns:
        Pipeline(): A Kedro pipeline object.
    """
    return Pipeline(
        [
            node(
                func=run_bte_queries,
                inputs={
                    "disease_list": "ingestion.raw.disease_list@pandas",
                    "drug_list": "ingestion.raw.drug_list@pandas",
                    "async_query_url": "params:bte.async_query_url",
                    "max_concurrent_requests": "params:bte.max_concurrent_requests",
                    "default_timeout": "params:bte.default_timeout",
                    "job_check_sleep": "params:bte.job_check_sleep",
                    "n_diseases_limit": "params:bte.n_diseases_limit",
                },
                outputs="bte.model_output.predictions",
                name="bte-single_processing_node",
            )
        ]
    )
