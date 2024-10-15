"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from matrix.pipelines.embeddings.pipeline import create_embeddings_pipeline
from matrix.pipelines.evaluation.pipeline import create_evaluation_pipeline
from matrix.pipelines.fabricator.pipeline import create_fabricator_pipeline
from matrix.pipelines.inference.pipeline import create_inference_pipeline
from matrix.pipelines.ingestion.pipeline import create_ingestion_pipeline
from matrix.pipelines.integration.pipeline import create_integration_pipeline
from matrix.pipelines.matrix_generation.pipeline import create_matrix_pipeline
from matrix.pipelines.modelling.pipeline import create_modelling_pipeline
from matrix.pipelines.preprocessing.pipeline import create_preprocessing_pipeline
from matrix.pipelines.release.pipeline import create_release_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        Mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipelines = {}

    pipelines["release"] = create_integration_pipeline() + create_embeddings_pipeline()
    pipelines["modelling"] = create_modelling_pipeline() + create_matrix_pipeline() + create_evaluation_pipeline()

    # TODO: What will we use?
    # pipelines["__default__"] = pipelines["integration_embeddings"] + pipelines["modelling_matrix_evaluation"]

    pipelines["test"] = (
        create_fabricator_pipeline()
        + create_ingestion_pipeline()
        + pipelines["integration_embeddings"]
        + pipelines["modelling_matrix_evaluation"]
        + create_release_pipeline()
    )

    # Ran manually based on input from medical to release new artifacts from clinical trails and medical KG
    pipelines["preprocessing"] = create_preprocessing_pipeline()

    # We only run whenever sources change
    pipelines["ingestion"] = create_ingestion_pipeline()

    # We run only manually based on medical input
    pipelines["inference"] = create_inference_pipeline()
    return pipelines
