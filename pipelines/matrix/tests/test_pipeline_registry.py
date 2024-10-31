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


# def test_default_pipeline_composition(pipelines: Dict[str, Pipeline]) -> None:
#     default_pipeline = pipelines["__default__"]
# <<<<<<< HEAD
#     release_pipeline = pipelines["data_release"]
#     ingestion_pipeline = pipelines["ingestion"]
#     preprocessing_pipeline = pipelines["preprocessing"]

#     # assert default  does do release
#     assert len(set(default_pipeline.nodes).intersection(set(release_pipeline.nodes))) != 0
#     # assert default does do ingestion
#     assert len(set(default_pipeline.nodes).intersection(set(ingestion_pipeline.nodes))) != 0
#     # assert preprocessing does not occur in default
#     assert len(set(default_pipeline.nodes).intersection(set(preprocessing_pipeline.nodes))) == 0
# =======
#     release_pipeline = pipelines["kg_release"]
#     embeddings_pipeline = pipelines["embeddings"]
#     modelling_pipeline = pipelines["modelling_run"]
#     pipelines_len = len(modelling_pipeline.nodes) + len(release_pipeline.nodes) + len(embeddings_pipeline.nodes)

#     assert (
#         len(default_pipeline.nodes) == pipelines_len
#     ), f"Default pipeline (len: {len(default_pipeline.nodes)}) should be the sum of release_pipeline (len: {len(release_pipeline.nodes)}) and make_modelling (len: {len(modelling_pipeline.nodes)}) and embeddings (len: {len(embeddings_pipeline.nodes)})"

#     assert default_pipeline.nodes == (release_pipeline + modelling_pipeline + embeddings_pipeline).nodes
# >>>>>>> main
