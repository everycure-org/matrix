"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from matrix.pipelines.preprocessing.pipeline import create_pipeline as create_preprocessing_pipeline
from matrix.pipelines.modelling.pipeline import create_pipeline as create_modelling_pipeline
from matrix.pipelines.fabricator.pipeline import create_pipeline as create_fabricator_pipeline
from matrix.pipelines.embeddings.pipeline import create_pipeline as create_embeddings_pipeline
from matrix.pipelines.integration.pipeline import create_pipeline as create_integration_pipeline
from matrix.pipelines.evaluation.pipeline import create_pipeline as create_evaluation_pipeline
from matrix.pipelines.ingestion.pipeline import create_pipeline as create_ingestion_pipeline
from matrix.pipelines.data_release.pipeline import create_pipeline as create_data_release_pipeline
from matrix.pipelines.matrix_generation.pipeline import create_pipeline as create_matrix_pipeline
from matrix.pipelines.inference.pipeline import create_pipeline as create_inference_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        Mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipelines = {}

    # TODO for now leaving embeddings out of KG release, we need to fix embeddings first for the large KG
    pipelines["kg_release"] = (
        create_ingestion_pipeline() + create_integration_pipeline() + create_data_release_pipeline()
    )  # + create_embeddings_pipeline()
    pipelines["modelling"] = create_modelling_pipeline() + create_matrix_pipeline() + create_evaluation_pipeline()

    # default doesn't do ingestion, enables e2e run locally
    pipelines["__default__"] = (
        create_integration_pipeline()
        + create_embeddings_pipeline()
        + create_modelling_pipeline()
        + create_matrix_pipeline()
        + create_evaluation_pipeline()
    )
    pipelines["test_release"] = (
        create_fabricator_pipeline()
        + create_ingestion_pipeline()
        + create_integration_pipeline()
        + create_embeddings_pipeline()
    )
    pipelines["test_modelling"] = (
        create_modelling_pipeline()
        + create_matrix_pipeline()
        + create_evaluation_pipeline()
        + create_data_release_pipeline()
    )

    # pipelines["release"] = pipelines["__default__"] + pipelines["ingestion"]  # + pipelines["preprocessing"]
    pipelines["test"] = (
        create_fabricator_pipeline()
        + create_ingestion_pipeline()
        + create_integration_pipeline()
        + create_embeddings_pipeline()
        + create_modelling_pipeline()
        + create_matrix_pipeline()
        + create_evaluation_pipeline()
        + create_data_release_pipeline()
    )

    # Ran manually based on input from medical to release new artifacts from clinical trails and medical KG
    pipelines["preprocessing"] = create_preprocessing_pipeline()

    # We only run whenever sources change
    pipelines["ingestion"] = create_ingestion_pipeline()

    # We run only manually based on medical input
    pipelines["inference"] = create_inference_pipeline()
    return pipelines
