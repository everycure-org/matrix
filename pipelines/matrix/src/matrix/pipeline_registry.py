"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from matrix.pipelines.data_release.pipeline import create_pipeline as create_data_release_pipeline
from matrix.pipelines.embeddings.pipeline import create_pipeline as create_embeddings_pipeline
from matrix.pipelines.evaluation.pipeline import create_pipeline as create_evaluation_pipeline
from matrix.pipelines.fabricator.pipeline import create_pipeline as create_fabricator_pipeline
from matrix.pipelines.inference.pipeline import create_pipeline as create_inference_pipeline
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
    # Create base pipelines
    data_release = create_data_release_pipeline()
    embeddings = create_embeddings_pipeline()
    evaluation = create_evaluation_pipeline()
    fabricator = create_fabricator_pipeline()
    inference = create_inference_pipeline()
    ingestion = create_ingestion_pipeline()
    integration = create_integration_pipeline()
    matrix_generation = create_matrix_pipeline()
    modelling = create_modelling_pipeline()
    preprocessing = create_preprocessing_pipeline()

    # Define pipeline combinations
    pipelines = {
        # Individual pipelines
        "data_release": data_release,
        "embeddings": embeddings,
        "evaluation": evaluation,
        "fabricator": fabricator,
        "inference": inference,  # Run manually based on medical input
        "ingestion": ingestion,  # Run only when sources change
        "integration": integration,
        "matrix_generation": matrix_generation,
        "preprocessing": preprocessing,  # Run manually for clinical trials and medical KG artifacts
        # Combined pipelines
        "kg_release": ingestion + integration + data_release,  # Embeddings temporarily excluded
        "modelling": modelling + matrix_generation + evaluation,
        "__default__": integration + embeddings + modelling + matrix_generation + evaluation,
        # Test pipelines
        "test_release": fabricator + ingestion + integration + embeddings,
        "test_modelling": modelling + matrix_generation + evaluation + data_release,
        "test": (
            fabricator
            + ingestion
            + integration
            + embeddings
            + modelling
            + matrix_generation
            + evaluation
            + data_release
        ),
    }

    return pipelines
