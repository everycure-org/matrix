"""Project pipelines."""

from kedro.pipeline import Pipeline

from review_list.pipelines.review_list.pipeline import (
    create_pipeline as create_review_list_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {"review_list": create_review_list_pipeline()}
    return pipelines
