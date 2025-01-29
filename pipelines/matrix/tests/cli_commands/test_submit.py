import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from click.testing import CliRunner
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from matrix.cli_commands.submit import (
    _submit,
    apply_argo_template,
    build_argo_template,
    build_push_docker,
    can_talk_to_kubernetes,
    command_exists,
    ensure_namespace,
    get_run_name,
    run_subprocess,
    save_argo_template,
    submit,
    submit_workflow,
)
from matrix.kedro4argo_node import ArgoResourceConfig


@pytest.fixture
def mock_run_subprocess():
    with patch("matrix.cli_commands.submit.run_subprocess") as mock:
        mock.return_value = MagicMock(stdout='{"metadata": {"name": "mocked-job-name"}}')
        yield mock


@pytest.fixture
def mock_dependencies():
    with patch("matrix.cli_commands.submit.can_talk_to_kubernetes") as _, patch(
        "matrix.cli_commands.submit.build_push_docker"
    ) as _, patch("matrix.cli_commands.submit.apply_argo_template") as _, patch(
        "matrix.cli_commands.submit.ensure_namespace"
    ):
        yield


@pytest.fixture(scope="function")
def mock_pipelines():
    pipeline_dict = {
        "__default__": MagicMock(),
        "mock_pipeline": MagicMock(),
        "mock_pipeline2": MagicMock(),
        "mock_pipeline3": MagicMock(),
    }

    with patch("matrix.cli_commands.submit.kedro_pipelines", new=pipeline_dict) as mock:
        yield mock


@pytest.fixture
def mock_submit_internal():
    with patch("matrix.cli_commands.submit._submit") as mock:
        yield mock


@pytest.fixture(scope="function")
def mock_multiple_pipelines():
    pipeline_dict = {
        "mock_pipeline": MagicMock(),
        "mock_pipeline2": MagicMock(),
        "mock_pipeline3": MagicMock(),
    }

    with patch("matrix.cli_commands.submit.kedro_pipelines", new=pipeline_dict) as mock:
        yield mock


@patch("matrix.cli_commands.submit.generate_argo_config")
def test_build_argo_template(mock_generate_argo_config: None) -> None:
    build_argo_template(
        "test_run", "testuser", "test_namespace", {"test": MagicMock()}, ArgoResourceConfig(), "cloud", is_test=True
    )
    mock_generate_argo_config.assert_called_once()


def test_ensure_namespace_existing(mock_run_subprocess: None) -> None:
    mock_run_subprocess.return_value.returncode = 0
    ensure_namespace("existing_namespace", verbose=True)
    assert mock_run_subprocess.call_count == 1


def test_ensure_namespace_new(mock_run_subprocess: None) -> None:
    mock_run_subprocess.side_effect = [MagicMock(returncode=1), MagicMock(returncode=0)]
    ensure_namespace("new_namespace", verbose=True)
    assert mock_run_subprocess.call_count == 2


@pytest.fixture()
def temporary_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_save_argo_template_creates_file(temporary_directory: Path) -> None:
    argo_template = "test template content"
    result = save_argo_template(argo_template, temporary_directory)

    assert Path(result).exists()
    assert Path(result).is_file()


def test_save_argo_template_content(temporary_directory: Path) -> None:
    argo_template = "test template content"

    file_path = save_argo_template(argo_template, temporary_directory)

    with open(file_path, "r") as f:
        content = f.read()

    assert content == argo_template


def test_save_argo_template_returns_string(temporary_directory: Path) -> None:
    argo_template = "test template content"

    result = save_argo_template(argo_template, temporary_directory)

    assert isinstance(result, str)


def test_apply_argo_template(mock_run_subprocess: None) -> None:
    apply_argo_template("test_namespace", Path("sample_template.yml"), verbose=True)
    mock_run_subprocess.assert_called_once()


def test_submit_workflow(mock_run_subprocess: None) -> None:
    mock_run_subprocess.return_value.stdout = '{"metadata": {"name": "test-job"}}'
    submit_workflow("test_run", "test_namespace", verbose=True)
    assert mock_run_subprocess.call_count == 1


@pytest.mark.parametrize(
    "input_name,expected_name",
    [
        ("custom_name", "custom-name"),
        ("custom-name", "custom-name"),
        ("custom@name!", "custom-name"),
    ],
)
def test_get_run_name_with_input(input_name: str, expected_name: str) -> None:
    assert expected_name in get_run_name(input_name)


@pytest.mark.skip(reason="Investigate why click is not correctly throwing up exceptions")
def test_pipeline_not_found(mock_multiple_pipelines):
    with pytest.raises(click.ClickException):
        # Given a CLI runner instance
        runner = CliRunner()

        # When invoking with non existing pipeline
        runner.invoke(submit, ["--username", "testuser", "--run-name", "test-run", "--pipeline", "not_exists"])


def test_command_exists(mock_run_subprocess: None) -> None:
    mock_run_subprocess.return_value.returncode = 0
    assert command_exists("existing_command") is True

    mock_run_subprocess.return_value.returncode = 1
    assert command_exists("non_existing_command") is False


def test_run_subprocess_error() -> None:
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_subprocess("invalid_command", stream_output=True)

    assert exc_info.value.returncode == 127
    assert exc_info.value.stdout is None


def test_run_subprocess_streaming() -> None:
    result = run_subprocess('echo "test"', stream_output=True)

    assert result.returncode == 0
    assert result.stdout == "test\n"
    assert result.stderr is None


def test_run_subprocess_no_streaming_2() -> None:
    result = run_subprocess('echo "test"', stream_output=False)

    assert result.returncode == 0
    assert result.stdout is None
    assert result.stderr is None


def test_run_subprocess_no_streaming_error() -> None:
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_subprocess("invalid_command", stream_output=False)

    assert exc_info.value.returncode == 127
    assert exc_info.value.stderr is None
    assert exc_info.value.stdout is None


@pytest.mark.parametrize("pipeline_for_execution", ["__default__", "test_pipeline"])
def test_workflow_submission(
    mock_run_subprocess: None, mock_dependencies: None, temporary_directory: Path, pipeline_for_execution: str
) -> None:
    def dummy_func(*args):
        """Dummy function for testing purposes."""
        return args

    pipeline_obj = Pipeline(
        nodes=[Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")]
    )
    pipeline_obj.name = pipeline_for_execution

    _submit(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        release_version="test_release",
        pipeline_obj=pipeline_obj,
        verbose=True,
        dry_run=False,
        template_directory=temporary_directory,
        allow_interactions=False,
        environment="cloud",
    )

    yaml_file = temporary_directory / "argo-workflow-template.yml"
    assert yaml_file.is_file(), f"Expected {yaml_file} to be a file"

    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    assert isinstance(content, dict), "Parsed YAML content should be a dictionary"

    templates = content.get("spec", {}).get("templates", [])
    pipeline_templates = [t for t in templates if "dag" in t]

    assert len(pipeline_templates) == 1, "Expected one pipeline template (test and cloud)"

    pipeline_names = [t["name"] for t in pipeline_templates]
    assert "pipeline" in pipeline_names, "Expected 'pipeline' pipeline to be present"

    # Additional checks
    assert content["metadata"]["name"] == "test-run", "Expected 'test-run' as the workflow name"
    assert content["metadata"]["namespace"] == "test_namespace", "Expected 'test_namespace' as the namespace"

    # Check for the presence of tasks in each pipeline
    for pipeline in pipeline_templates:
        tasks = pipeline.get("dag", {}).get("tasks", [])
        assert len(tasks) > 0, f"Expected at least one task in the {pipeline['name']} pipeline"
    submit_cmd = " ".join(
        [
            "argo submit",
            "--name test-run",
            "-n test_namespace",
            "--from wftmpl/test-run",
            "-p run_name=test-run",
            "-l submit-from-ui=false",
            "-o json",
        ]
    )

    mock_run_subprocess.assert_called_with(submit_cmd)
