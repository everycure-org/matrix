"""Project pipelines."""

from kedro.pipeline import Pipeline  # type: ignore[import-not-found]

from huggingface_dataset_demo.pipelines.kg_edges import create_pipeline as kg_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register only the KG edges pipeline as default."""
    kg = kg_pipeline()
    return {
        "kg_edges": kg,
        "__default__": kg,
    }
