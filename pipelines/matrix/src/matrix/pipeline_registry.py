"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline

from matrix.pipelines.modelling.pipeline import (
    create_pipeline as create_modelling_pipeline,
)
from matrix.pipelines.embeddings.pipeline import (
    create_pipeline as create_embeddings_pipeline,
)
from matrix.pipelines.integration.pipeline import (
    create_pipeline as create_integration_pipeline,
)
from matrix.pipelines.evaluation.pipeline import (
    create_pipeline as create_evaluation_pipeline,
)
from matrix.pipelines.ingestion.pipeline import (
    create_pipeline as create_ingestion_pipeline,
)
from matrix.pipelines.matrix_generation.pipeline import (
    create_pipeline as create_matrix_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        Mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipelines = {}

    pipelines["embeddings"] = create_integration_pipeline() + create_embeddings_pipeline()
    pipelines["modelling"] = create_modelling_pipeline() + create_matrix_pipeline() + create_evaluation_pipeline()
    pipelines["full"] = create_modelling_pipeline() + create_matrix_pipeline() + create_evaluation_pipeline()
    pipelines["__default__"] = pipelines["embeddings"] + pipelines["modelling"]
    pipelines["all"] = create_ingestion_pipeline() + pipelines["__default__"]
    return pipelines
