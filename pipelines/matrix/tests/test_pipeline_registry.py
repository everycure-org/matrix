import pytest
from kedro.pipeline import Pipeline
from matrix.pipeline_registry import register_pipelines


@pytest.fixture
def pipelines():
    return register_pipelines()


def test_register_pipelines_returns_dict(pipelines):
    assert isinstance(pipelines, dict)


def test_all_pipeline_names_present(pipelines):
    expected_pipelines = [
        "make_embeddings",
        "make_modelling",
        "__default__",
        "preprocessing",
        "ingestion",
        "modelling",
        "embeddings",
        "fabricator",
        "integration",
        "evaluation",
        "release",
        "matrix_generation",
        "test",
        "all",
        "experiment",
    ]
    for pipeline_name in expected_pipelines:
        assert pipeline_name in pipelines


def test_all_pipelines_are_pipeline_objects(pipelines):
    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)


def test_default_pipeline_composition(pipelines):
    default_pipeline = pipelines["__default__"]
    make_embeddings = pipelines["make_embeddings"]
    make_modelling = pipelines["make_modelling"]

    assert (
        len(default_pipeline.nodes)
        == len(make_embeddings.nodes) + len(make_modelling.nodes)
    ), f"Default pipeline (len: {len(default_pipeline.nodes)}) should be the sum of make_embeddings (len: {len(make_embeddings.nodes)}) and make_modelling (len: {len(make_modelling.nodes)})"
    assert default_pipeline.nodes == (make_embeddings + make_modelling).nodes


def test_make_embeddings_composition(pipelines):
    make_embeddings = pipelines["make_embeddings"]
    integration = pipelines["integration"]
    embeddings = pipelines["embeddings"]

    assert make_embeddings == integration + embeddings


def test_make_modelling_composition(pipelines):
    make_modelling = pipelines["make_modelling"]
    modelling = pipelines["modelling"]
    matrix_generation = pipelines["matrix_generation"]
    evaluation = pipelines["evaluation"]

    assert make_modelling == modelling + matrix_generation + evaluation


def test_test_pipeline_composition(pipelines):
    test_pipeline = pipelines["test"]
    fabricator = pipelines["fabricator"]
    ingestion = pipelines["ingestion"]
    make_embeddings = pipelines["make_embeddings"]
    make_modelling = pipelines["make_modelling"]
    release = pipelines["release"]

    assert (
        test_pipeline
        == fabricator + ingestion + make_embeddings + make_modelling + release
    )


def test_all_pipeline_composition(pipelines):
    all_pipeline = pipelines["all"]
    ingestion = pipelines["ingestion"]
    default = pipelines["__default__"]

    assert all_pipeline == ingestion + default


def test_experiment_pipeline_composition(pipelines):
    experiment_pipeline = pipelines["experiment"]
    modelling = pipelines["modelling"]
    evaluation = pipelines["evaluation"]

    assert experiment_pipeline == modelling + evaluation
