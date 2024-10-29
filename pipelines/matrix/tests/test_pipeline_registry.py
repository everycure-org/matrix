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
        "__default__",
    ]
    for pipeline_name in expected_pipelines:
        assert pipeline_name in pipelines


def test_all_pipelines_are_pipeline_objects(pipelines: Dict[str, Pipeline]) -> None:
    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)


def test_default_pipeline_composition(pipelines: Dict[str, Pipeline]) -> None:
    default_pipeline = pipelines["__default__"]
    release_pipeline = pipelines["release"]
    ingestion_pipeline = pipelines["ingestion"]

    # assert default does not do release
    assert len(set(default_pipeline.nodes).intersection(set(release_pipeline.nodes))) == 0
    # assert default does not do ingestion
    assert len(set(default_pipeline.nodes).intersection(set(ingestion_pipeline.nodes))) == 0
