from typing import Dict
import pytest
from kedro.pipeline import Pipeline
from matrix.pipeline_registry import register_pipelines


@pytest.fixture
def pipelines() -> Dict[str, Pipeline]:
    return register_pipelines()


def test_register_pipelines_returns_dict(pipelines: Dict[str, Pipeline]) -> None:
    assert isinstance(pipelines, dict)


def test_all_pipeline_names_present(pipelines: Dict[str, Pipeline]) -> None:
    expected_pipelines = [
        "kg_release",
        "modelling",
        "default",
    ]
    for pipeline_name in expected_pipelines:
        assert pipeline_name in pipelines


def test_all_pipelines_are_pipeline_objects(pipelines: Dict[str, Pipeline]) -> None:
    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)


def test_default_pipeline_composition(pipelines: Dict[str, Pipeline]) -> None:
    default_pipeline = pipelines["default"]
    release_pipeline = pipelines["data_release"]
    ingestion_pipeline = pipelines["ingestion"]
    preprocessing_pipeline = pipelines["preprocessing"]

    # assert default  does do release
    assert len(set(default_pipeline.nodes).intersection(set(release_pipeline.nodes))) != 0
    # assert default does do ingestion
    assert len(set(default_pipeline.nodes).intersection(set(ingestion_pipeline.nodes))) != 0
    # assert preprocessing does not occur in default
    assert len(set(default_pipeline.nodes).intersection(set(preprocessing_pipeline.nodes))) == 0
