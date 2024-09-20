"""This module creates a pipeline for the BTE process."""

from kedro.pipeline import Pipeline, node
from .async_query_processing import bte_kedro_node_function


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for the BTE process.

    :param kwargs: Additional keyword arguments.
    :return: A Kedro pipeline object.
    """
    return Pipeline(
        [
            node(
                func=bte_kedro_node_function,
                # inputs="ingestion.raw.disease_list@pandas",
                inputs=None,
                outputs="modelling.bte_model.model_output.predictions",
                name="bte-single_processing_node",
            )
        ]
    )
