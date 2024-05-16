"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from matrix.pipelines.modeling.pipeline import (
    create_pipeline as create_modeling_pipeline,
)
from matrix.pipelines.fabricator.pipeline import (
    create_pipeline as create_fabricator_pipeline,
)
from matrix.pipelines.embeddings.pipeline import (
    create_pipeline as create_embeddings_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines["__default__"] = create_modeling_pipeline()
    pipelines["embeddings"] = create_embeddings_pipeline()
    pipelines["fabricator"] = create_fabricator_pipeline()
    return pipelines
