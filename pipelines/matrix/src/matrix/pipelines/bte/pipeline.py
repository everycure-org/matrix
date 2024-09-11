"""This module creates a pipeline for the BTE process."""

from kedro.pipeline import Pipeline, node

# from .tsv_ingestion import ingest_tsv
from .query_generation import generate_queries
from .async_query_processing import fetch_results
from .result_transformation import transform_results


def create_pipeline(**kwargs) -> Pipeline:
    """Creates a Kedro pipeline for the BTE process.

    :param kwargs: Additional keyword arguments.
    :return: A Kedro pipeline object.
    """
    return Pipeline(
        [
            node(
                func=generate_queries,
                inputs=None,
                outputs="bte-queries",
                name="bte_generate_queries_node",
            ),
            node(
                func=fetch_results,
                inputs="bte-queries",
                outputs="bte-raw_results",
                name="bte_fetch_results_node",
            ),
            node(
                func=transform_results,
                inputs="bte-raw_results",
                outputs="bte-final_dataframe",
                name="bte_transform_results_node",
            ),
        ]
    )
