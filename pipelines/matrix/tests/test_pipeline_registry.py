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

    assert (
        len(make_embeddings.nodes) == len(integration.nodes) + len(embeddings.nodes)
    ), f"Default pipeline (len: {len(make_embeddings.nodes)}) should be the sum of integration (len: {len(integration.nodes)}) and embeddings (len: {len(embeddings.nodes)})"

    assert make_embeddings.nodes == (integration + embeddings).nodes


def test_make_modelling_composition(pipelines):
    make_modelling = pipelines["make_modelling"]
    modelling = pipelines["modelling"]
    matrix_generation = pipelines["matrix_generation"]
    evaluation = pipelines["evaluation"]

    assert (
        len(make_modelling.nodes)
        == len(modelling.nodes) + len(matrix_generation.nodes) + len(evaluation.nodes)
    ), f"Make modelling pipeline (len: {len(make_modelling.nodes)}) should be the sum of modelling (len: {len(modelling.nodes)}), matrix_generation (len: {len(matrix_generation.nodes)}) and evaluation (len: {len(evaluation.nodes)})"
    assert make_modelling.nodes == (modelling + matrix_generation + evaluation).nodes


def test_test_pipeline_composition(pipelines):
    test_pipeline = pipelines["test"]
    fabricator = pipelines["fabricator"]
    ingestion = pipelines["ingestion"]
    make_embeddings = pipelines["make_embeddings"]
    make_modelling = pipelines["make_modelling"]
    release = pipelines["release"]

    assert (
        len(test_pipeline.nodes)
        == len(fabricator.nodes)
        + len(ingestion.nodes)
        + len(make_embeddings.nodes)
        + len(make_modelling.nodes)
        + len(release.nodes)
    ), f"Test pipeline (len: {len(test_pipeline.nodes)}) should be the sum of fabricator (len: {len(fabricator.nodes)}), ingestion (len: {len(ingestion.nodes)}), make_embeddings (len: {len(make_embeddings.nodes)}), make_modelling (len: {len(make_modelling.nodes)}) and release (len: {len(release.nodes)})"
    assert (
        test_pipeline.nodes
        == (fabricator + ingestion + make_embeddings + make_modelling + release).nodes
    )


def test_all_pipeline_composition(pipelines):
    all_pipeline = pipelines["all"]
    ingestion = pipelines["ingestion"]
    default = pipelines["__default__"]

    assert (
        len(all_pipeline.nodes) == len(ingestion.nodes) + len(default.nodes)
    ), f"All pipeline (len: {len(all_pipeline.nodes)}) should be the sum of ingestion (len: {len(ingestion.nodes)}) and default (len: {len(default.nodes)})"
    assert all_pipeline.nodes == (ingestion + default).nodes


def test_experiment_pipeline_composition(pipelines):
    experiment_pipeline = pipelines["experiment"]
    modelling = pipelines["modelling"]
    evaluation = pipelines["evaluation"]

    assert (
        len(experiment_pipeline.nodes) == len(modelling.nodes) + len(evaluation.nodes)
    ), f"Experiment pipeline (len: {len(experiment_pipeline.nodes)}) should be the sum of modelling (len: {len(modelling.nodes)}) and evaluation (len: {len(evaluation.nodes)})"
    assert experiment_pipeline.nodes == (modelling + evaluation).nodes
