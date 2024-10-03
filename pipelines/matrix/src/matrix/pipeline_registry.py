"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from matrix.pipelines.preprocessing.pipeline import (
    create_pipeline as create_preprocessing_pipeline,
)
from matrix.pipelines.modelling.pipeline import (
    create_pipeline as create_modelling_pipeline,
)
from matrix.pipelines.fabricator.pipeline import (
    create_pipeline as create_fabricator_pipeline,
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
from matrix.pipelines.preprocessing.pipeline import (
    create_pipeline as create_preprocessing_pipeline,
)
from matrix.pipelines.release.pipeline import (
    create_pipeline as create_release_pipeline,
)
from matrix.pipelines.matrix_generation.pipeline import (
    create_pipeline as create_matrix_pipeline,
)
from matrix.pipelines.inference.pipeline import (
    create_pipeline as create_inference_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        Mapping from a pipeline name to a ``Pipeline`` object.
    """
    pipelines = {}

    pipelines["make_embeddings"] = (
        create_integration_pipeline() + create_embeddings_pipeline()
    )
    pipelines["make_modelling"] = (
        create_modelling_pipeline()
        + create_matrix_pipeline()
        + create_evaluation_pipeline()
    )
    pipelines["__default__"] = (
        pipelines["make_embeddings"] + pipelines["make_modelling"]
    )
    pipelines["preprocessing"] = create_preprocessing_pipeline()
    pipelines["ingestion"] = create_ingestion_pipeline()
    pipelines["preprocessing"] = create_preprocessing_pipeline()
    pipelines["modelling"] = create_modelling_pipeline()
    pipelines["embeddings"] = create_embeddings_pipeline()
    pipelines["fabricator"] = create_fabricator_pipeline()
    pipelines["integration"] = create_integration_pipeline()
    pipelines["evaluation"] = create_evaluation_pipeline()
    pipelines["release"] = create_release_pipeline()
    pipelines["matrix_generation"] = create_matrix_pipeline()
    pipelines["inference"] = create_inference_pipeline()
    pipelines["test"] = (
        create_fabricator_pipeline()
        + create_ingestion_pipeline()
        + pipelines["make_embeddings"]
        + pipelines["make_modelling"]
        + create_release_pipeline()
    )
    pipelines["all"] = create_ingestion_pipeline() + pipelines["__default__"]
    pipelines["experiment"] = create_modelling_pipeline() + create_evaluation_pipeline()
    return pipelines
