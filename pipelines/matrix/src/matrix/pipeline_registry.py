from typing import Dict

from kedro.pipeline import Pipeline

from matrix.pipelines.create_sample.pipeline import create_pipeline as create_create_sample_pipeline
from matrix.pipelines.data_release.pipeline import create_pipeline as create_data_release_pipeline
from matrix.pipelines.embeddings.pipeline import create_pipeline as create_embeddings_pipeline
from matrix.pipelines.evaluation.pipeline import create_pipeline as create_evaluation_pipeline
from matrix.pipelines.fabricator.pipeline import create_pipeline as create_fabricator_pipeline
from matrix.pipelines.ingestion.pipeline import create_pipeline as create_ingestion_pipeline
from matrix.pipelines.integration.pipeline import create_pipeline as create_integration_pipeline
from matrix.pipelines.matrix_generation.pipeline import create_pipeline as create_matrix_pipeline
from matrix.pipelines.modelling.pipeline import create_pipeline as create_modelling_pipeline
from matrix.pipelines.preprocessing.pipeline import create_pipeline as create_preprocessing_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        Mapping from a pipeline name to a ``Pipeline`` object.
    """
    # Define pipeline combinations
    pipelines = {
        # Individual pipelines
        "preprocessing": create_preprocessing_pipeline(),  # Run manually for clinical trials and medical KG artifacts
        "fabricator": create_fabricator_pipeline(),
        "ingestion": create_ingestion_pipeline(),
        "integration": create_integration_pipeline(),
        "embeddings": create_embeddings_pipeline(),
        "data_release": create_data_release_pipeline(),
        "modelling": create_modelling_pipeline(),
        "matrix_generation": create_matrix_pipeline(),
        "evaluation": create_evaluation_pipeline(),
        "create_sample": create_create_sample_pipeline(),
        # "inference": create_inference_pipeline(),  # Run manually based on medical input
    }

    # Higher order pipelines
    # fmt: off
    pipelines["data_engineering"] = (
          pipelines["ingestion"]
        + pipelines["integration"]
        + pipelines["embeddings"]
    )
    pipelines["kg_release"] = (
        pipelines["data_engineering"]
        + pipelines["data_release"] 
    )
    pipelines["modelling_run"] = (
          pipelines["modelling"]
        + pipelines["matrix_generation"]
        + pipelines["evaluation"]
    )
    pipelines["__default__"] = (
          pipelines["data_engineering"]
        + pipelines["modelling_run"]
    )

    # Test pipelines
    pipelines["test"] = (
        pipelines["fabricator"]
        + pipelines["__default__"]
        + pipelines["data_release"] 
    )
    pipelines["test_sample"] = (
        pipelines["embeddings"]
        + pipelines["modelling_run"]
    )
    # fmt: on

    return pipelines
