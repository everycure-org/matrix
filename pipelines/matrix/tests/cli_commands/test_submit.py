import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, _patch

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
    check_dependencies,
    command_exists,
    ensure_namespace,
    get_run_name,
    run_subprocess,
    save_argo_template,
    submit,
    submit_workflow,
)
from matrix.cli_commands import submit as mot_submit
from matrix.kedro4argo_node import ArgoResourceConfig


@pytest.fixture
def mock_run_subprocess():
    with patch("matrix.cli_commands.submit.run_subprocess") as mock:
        mock.return_value = MagicMock(stdout='{"metadata": {"name": "mocked-job-name"}}')
        yield mock


@pytest.fixture
def mock_dependencies():
    with patch("matrix.cli_commands.submit.check_dependencies") as _, patch(
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


def test_check_dependencies(mock_run_subprocess: None) -> None:
    mock_run_subprocess.return_value.returncode = 0
    mock_run_subprocess.return_value.stdout = "active_account"
    check_dependencies(verbose=True)
    assert mock_run_subprocess.call_count > 0


def test_build_push_docker(mock_run_subprocess: None) -> None:
    build_push_docker("testuser", verbose=True)
    mock_run_subprocess.assert_called_once_with("make docker_push TAG=testuser", stream_output=True)


@patch("matrix.cli_commands.submit.generate_argo_config")
def test_build_argo_template(mock_generate_argo_config: None) -> None:
    build_argo_template(
        run_name="test_run",
        username="testuser",
        namespace="test_namespace",
        pipeline_obj={"test": MagicMock()},
        default_execution_resources=ArgoResourceConfig(),
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


@pytest.fixture
def mock_popen():
    with patch("subprocess.Popen") as mock:
        yield mock


def test_run_subprocess_success(mock_popen: None) -> None:
    # Mock successful command execution
    process_mock = MagicMock()
    process_mock.stdout = iter(["output line 1\n", "output line 2\n"])
    process_mock.stderr = iter([""])
    process_mock.wait.return_value = 0
    mock_popen.return_value = process_mock

    result = run_subprocess('echo "test"', stream_output=True)

    assert result.returncode == 0
    assert result.stdout == "output line 1\noutput line 2\n"
    assert result.stderr == ""


def test_run_subprocess_error(mock_popen: None) -> None:
    # Mock command execution with error
    process_mock = MagicMock()
    process_mock.stdout = iter([""])
    process_mock.stderr = iter(["error message\n"])
    process_mock.wait.return_value = 1
    mock_popen.return_value = process_mock

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_subprocess("invalid_command", stream_output=True)

    assert exc_info.value.returncode == 1
    assert "error message" in exc_info.value.stderr


def test_run_subprocess_no_streaming(mock_popen: None):
    # Mock subprocess.run for non-streaming case
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args='echo "test"', returncode=0, stdout="output", stderr=""
        )

        result = run_subprocess('echo "test"', stream_output=False)

        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""


def test_run_subprocess_no_streaming_error(mock_popen: None) -> None:
    # Mock subprocess.run for non-streaming case with error
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test", output="", stderr="error")

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            run_subprocess("invalid_command", stream_output=False)

        assert exc_info.value.returncode == 1
        assert "error" in exc_info.value.stderr


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
    )

    yaml_files = list(temporary_directory.glob("argo-workflow-template.yml"))
    assert len(yaml_files) == 1, f"Expected 1 YAML file, found {len(yaml_files)}"

    yaml_file = yaml_files[0]
    assert yaml_file.is_file(), f"Expected {yaml_file} to be a file"
    assert yaml_file.name.endswith(".yml"), f"File does not have .yml extension: {yaml_file.name}"

    # Read and parse the YAML file
    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    # Check if the content is a dictionary
    assert isinstance(content, dict), "Parsed YAML content should be a dictionary"

    # Check for the presence of two pipelines in the templates
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
            "--namespace test_namespace",
            "--from wftmpl/test-run",
            "--parameter run_name=test-run",
            "--labels submit-from-ui=false",
            "--output json",
        ]
    )

    mock_run_subprocess.assert_called_with(submit_cmd, capture_output=True, stream_output=True)


def test_release_exists(mock_run_subprocess: _patch):
    mock_run_subprocess.return_value.stdout = "8d5eb6af9af52eb31134ac1ec346c4e7a695c228        refs/tags/v0.2"
    assert mot_submit.release_exists("v1.0.0") is False
    assert mot_submit.release_exists("v0.2") is True


def test_release_exists_bad_input():
    with pytest.raises(AssertionError):
        mot_submit.release_exists("version")


def test_release_exists_from_non_git_folder():
    os.chdir(tempfile.gettempdir())
    assert mot_submit.release_exists("v1.nonexisting") is False
