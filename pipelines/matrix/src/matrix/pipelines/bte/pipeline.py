"""This module creates a pipeline for the BTE process."""

from kedro.pipeline import Pipeline, node
from .async_query_processing import run_bte_queries


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for the BTE process.

    :param kwargs: Additional keyword arguments.
    :return: A Kedro pipeline object.
    """
    return Pipeline(
        [
            node(
                func=run_bte_queries,
                # inputs="ingestion.raw.disease_list@pandas",
                inputs=None,
                outputs="modelling.bte_model.model_output.predictions",
                name="bte-single_processing_node",
            )
        ]
    )
