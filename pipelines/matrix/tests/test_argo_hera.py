"""
# NOTE: This file was partially generated using AI assistance.

Tests for Hera-based Argo Workflow generation.
"""

from __future__ import annotations
# ruff: noqa: I001

from pathlib import Path
import subprocess
import shutil

from kedro.pipeline import Pipeline  # type: ignore[import]
from kedro.pipeline.node import Node  # type: ignore[import]
import pytest  # type: ignore[import]
import yaml  # type: ignore[import]

from matrix.argo_hera import build_workflow_yaml
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig


def _run_subprocess(cmd: str):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def dummy_fn(*args):
    return "ok"


@pytest.fixture()
def complex_pipeline() -> Pipeline:
    # Build a pipeline with fan-in/fan-out and mixed ArgoResourceConfig
    n1 = ArgoNode(
        func=dummy_fn,
        inputs=["a", "b"],
        outputs="c",
        name="n1",
        argo_config=ArgoResourceConfig(cpu_request=2, cpu_limit=4, memory_request=16, memory_limit=32, num_gpus=0),
        tags=["argowf.fuse", "argowf.fuse-group.stage1"],
    )
    n2 = ArgoNode(
        func=dummy_fn,
        inputs="c",
        outputs="d",
        name="n2",
        argo_config=ArgoResourceConfig(cpu_request=4, cpu_limit=8, memory_request=32, memory_limit=64, num_gpus=1),
        tags=["argowf.fuse", "argowf.fuse-group.stage1"],
    )
    n3 = ArgoNode(
        func=dummy_fn,
        inputs="d",
        outputs="e",
        name="n3",
        tags=["argowf.fuse", "argowf.fuse-group.stage2"],
    )
    n4 = Node(func=dummy_fn, inputs=["e"], outputs="f", name="n4")
    n5 = Node(func=dummy_fn, inputs=["e"], outputs="g", name="n5")
    n6 = Node(func=dummy_fn, inputs=["f", "g"], outputs="h", name="n6")

    p = Pipeline(nodes=[n1, n2, n3, n4, n5, n6])
    p.name = "complex"
    return p


def test_build_workflow_yaml_and_lint(tmp_path: Path, complex_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch):
    # Minimal env expected by the builder
    monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", "0")
    monkeypatch.setenv("RUNTIME_GCP_PROJECT_ID", "ec-dev-project")
    # Build YAML
    wf_yaml = build_workflow_yaml(
        image="us-central1-docker.pkg.dev/dev/matrix:unit",
        run_name="unit-run",
        release_version="v0.0.0-test",
        mlflow_experiment_id=123,
        mlflow_url="https://mlflow.local/",
        namespace="argo-workflows",
        username="tester",
        pipeline=complex_pipeline,
        environment="cloud",
        package_name="matrix",
        release_folder_name="releases",
        default_execution_resources=ArgoResourceConfig(),
        mlflow_run_id="mlflow-run-id",
    )

    # Basic YAML parse and structural checks
    wf = yaml.safe_load(wf_yaml)
    assert wf["kind"] == "Workflow"
    assert wf["metadata"]["name"] == "unit-run"
    assert wf["spec"]["entrypoint"] == "main"

    # Persist and lint via argo (workflow lint). Skip if argo CLI is not present.
    if shutil.which("argo") is None:
        pytest.skip("argo CLI not installed; skipping lint check")
    f = tmp_path / "wf.yaml"
    f.write_text(wf_yaml)
    result = _run_subprocess(f"argo lint {f}")
    assert result.returncode == 0


def _get_templates_by_kind(wf_dict: dict) -> tuple[list[dict], list[dict], dict]:
    """Helper to split templates by kind and return (containers, dags, by_name)."""
    templates = wf_dict["spec"]["templates"]
    containers = [t for t in templates if "container" in t]
    dags = [t for t in templates if "dag" in t]
    by_name = {t["name"]: t for t in templates}
    return containers, dags, by_name


@pytest.mark.usefixtures("complex_pipeline")
def test_fusing_and_template_counts_non_neo4j(complex_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch):
    # Given: no neo4j tag -> expect 1 DAG + 1 container template
    monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", "0")
    monkeypatch.setenv("RUNTIME_GCP_PROJECT_ID", "ec-dev-project")
    complex_pipeline.name = "complex"

    wf_yaml = build_workflow_yaml(
        image="us-central1-docker.pkg.dev/dev/matrix:unit",
        run_name="unit-run",
        release_version="v0.0.0-test",
        mlflow_experiment_id=123,
        mlflow_url="https://mlflow.local/",
        namespace="argo-workflows",
        username="tester",
        pipeline=complex_pipeline,
        environment="cloud",
        package_name="matrix",
        release_folder_name="releases",
        default_execution_resources=ArgoResourceConfig(),
        mlflow_run_id="mlflow-run-id",
    )

    wf = yaml.safe_load(wf_yaml)
    containers, dags, by_name = _get_templates_by_kind(wf)

    # Then: exactly 1 DAG (main) and 2 container templates defined (kedro-task, kedro-task-neo4j)
    assert len(dags) == 1
    assert dags[0]["name"] == "main"
    assert len(containers) == 2
    cont_names = {c["name"] for c in containers}
    assert {"kedro-task", "kedro-task-neo4j"} == cont_names

    # And: fusing collapsed n1,n2 into a single task whose kedro_nodes argument includes both
    tasks = by_name["main"]["dag"]["tasks"]
    # number of tasks equals 5 (n1+n2 fused, n3, n4, n5, n6)
    assert len(tasks) == 5
    kedro_nodes_args = {
        t["name"]: next(p for p in t.get("arguments", {}).get("parameters", []) if p["name"] == "kedro_nodes")["value"]
        for t in tasks
    }
    # One of the tasks should be the fused stage (contains both n1 and n2)
    assert any("," in v and set(v.split(",")) >= {"n1", "n2"} for v in kedro_nodes_args.values())


def _make_pipeline_with_neo4j() -> Pipeline:
    # Given: a simple chain with one neo4j-tagged node
    n1 = ArgoNode(
        func=dummy_fn,
        inputs=["a"],
        outputs="b",
        name="neo",
        argo_config=ArgoResourceConfig(cpu_request=2, cpu_limit=4, memory_request=16, memory_limit=32, num_gpus=0),
        tags=["argowf.fuse", "argowf.template-neo4j"],
    )
    n2 = Node(func=dummy_fn, inputs="b", outputs="c", name="plain")
    p = Pipeline(nodes=[n1, n2])
    p.name = "neo-pipe"
    return p


def test_neo4j_template_and_arguments(monkeypatch: pytest.MonkeyPatch):
    # When: building YAML with a neo4j-tagged node
    monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", "0")
    monkeypatch.setenv("RUNTIME_GCP_PROJECT_ID", "ec-dev-project")
    pipe = _make_pipeline_with_neo4j()
    wf_yaml = build_workflow_yaml(
        image="us-central1-docker.pkg.dev/dev/matrix:unit",
        run_name="unit-run",
        release_version="v0.0.0-test",
        mlflow_experiment_id=1,
        mlflow_url="https://mlflow.local/",
        namespace="argo-workflows",
        username="tester",
        pipeline=pipe,
        environment="cloud",
        package_name="matrix",
        release_folder_name="releases",
        default_execution_resources=ArgoResourceConfig(),
        mlflow_run_id="mlflow-run-id",
    )

    # Then: expect both default and neo4j container templates present
    wf = yaml.safe_load(wf_yaml)
    containers, dags, by_name = _get_templates_by_kind(wf)
    names = {t["name"] for t in containers}
    assert {"kedro-task", "kedro-task-neo4j"} <= names

    # And: at least one DAG task references the neo4j template and has neo4j_heap_g argument
    tasks = by_name["main"]["dag"]["tasks"]
    neo_tasks = [t for t in tasks if t["template"] == "kedro-task-neo4j"]
    assert len(neo_tasks) == 1
    neo_args = {p["name"]: p["value"] for p in neo_tasks[0]["arguments"]["parameters"]}
    assert "neo4j_heap_g" in neo_args


def test_per_task_resources_are_parameterized(complex_pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch):
    # Given: complex pipeline with different resource configs for early nodes
    monkeypatch.setenv("INCLUDE_PRIVATE_DATASETS", "0")
    monkeypatch.setenv("RUNTIME_GCP_PROJECT_ID", "ec-dev-project")
    complex_pipeline.name = "complex"
    wf_yaml = build_workflow_yaml(
        image="us-central1-docker.pkg.dev/dev/matrix:unit",
        run_name="unit-run",
        release_version="v0.0.0-test",
        mlflow_experiment_id=1,
        mlflow_url="https://mlflow.local/",
        namespace="argo-workflows",
        username="tester",
        pipeline=complex_pipeline,
        environment="cloud",
        package_name="matrix",
        release_folder_name="releases",
        default_execution_resources=ArgoResourceConfig(),
        mlflow_run_id="mlflow-run-id",
    )

    wf = yaml.safe_load(wf_yaml)
    tasks = next(t for t in wf["spec"]["templates"] if t.get("name") == "main")["dag"]["tasks"]

    # Extract resource arguments for each task
    def argmap(t):
        return {p["name"]: p["value"] for p in t.get("arguments", {}).get("parameters", [])}

    args_by_task = {t["name"]: argmap(t) for t in tasks}

    # Then: find fused stage (n1,n2) and ensure it carries the max resources of its fused nodes (n2 has GPU=1)
    fused_task_name = next(k for k, v in args_by_task.items() if "," in v.get("kedro_nodes", ""))
    fused_args = args_by_task[fused_task_name]
    # From fixture: n1 cpu_limit=4, n2 cpu_limit=8 => expect 8; n2 has num_gpus=1
    assert fused_args["cpu_limit"] == "8"
    assert fused_args["num_gpus"] == "1"
