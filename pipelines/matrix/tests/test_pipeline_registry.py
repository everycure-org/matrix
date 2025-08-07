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
        "kg_release_and_matrix_run",
        "kg_release_patch_and_matrix_run",
        "data_release",
        "feature_and_modelling_run",
        "__default__",
    ]
    for pipeline_name in expected_pipelines:
        assert pipeline_name in pipelines


def test_all_pipelines_are_pipeline_objects(pipelines: Dict[str, Pipeline]) -> None:
    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)


def test_default_pipeline_composition(pipelines: Dict[str, Pipeline]) -> None:
    default_pipeline = pipelines["__default__"]
    release_pipeline = pipelines["data_release"]
    ingestion_pipeline = pipelines["ingestion"]
    preprocessing_pipeline = pipelines["preprocessing"]

    nodes_of_default = set(default_pipeline.nodes)
    # assert default does not do release
    assert nodes_of_default.intersection(set(release_pipeline.nodes)) == set()
    # assert default does do ingestion
    assert len(nodes_of_default.intersection(set(ingestion_pipeline.nodes))) != 0
    # assert preprocessing does not occur in default
    assert len(nodes_of_default.intersection(set(preprocessing_pipeline.nodes))) == 0
