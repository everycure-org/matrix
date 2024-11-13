from pathlib import Path
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import yaml
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
from matrix.argo import ARGO_TEMPLATES_DIR_PATH
from matrix.cli_commands.submit import (
    _submit,
    save_argo_template,
    submit,
    check_dependencies,
    build_push_docker,
    build_argo_template,
    ensure_namespace,
    apply_argo_template,
    submit_workflow,
    get_run_name,
    command_exists,
    run_subprocess,
)
import subprocess

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


@pytest.fixture
def mock_submit_internal():
    with patch("matrix.cli_commands.submit._submit") as mock:
        yield mock


@pytest.fixture(scope="function")
def mock_pipelines():
    pipeline_dict = {
        "__default__": MagicMock(),
    }

    with patch("matrix.cli_commands.submit.kedro_pipelines", new=pipeline_dict) as mock:
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


def test_submit_simple(mock_submit_internal: None, mock_pipelines: None) -> None:
    runner = CliRunner()
    result = runner.invoke(submit, ["--username", "testuser", "--run-name", "test-run"])
    assert result.exit_code == 0

    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="argo-workflows",
        run_name="test-run",
        pipelines_for_workflow=mock_pipelines,
        pipeline_for_execution="__default__",
        verbose=False,
        dry_run=False,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_namespace(mock_pipelines: None, mock_submit_internal: None):
    runner = CliRunner()
    result = runner.invoke(
        submit, ["--username", "testuser", "--namespace", "test_namespace", "--run-name", "test-run"]
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow=mock_pipelines,
        pipeline_for_execution="__default__",
        verbose=False,
        dry_run=False,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_pipelines(mock_multiple_pipelines: None, mock_submit_internal: None):
    runner = CliRunner()
    result = runner.invoke(
        submit,
        [
            "--username",
            "testuser",
            "--namespace",
            "test_namespace",
            "--run-name",
            "test-run",
            "--pipeline",
            "mock_pipeline2",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow=mock_multiple_pipelines,
        pipeline_for_execution="mock_pipeline2",
        verbose=False,
        dry_run=False,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_multiple_pipelines(mock_multiple_pipelines: None, mock_submit_internal: None):
    runner = CliRunner()
    result = runner.invoke(
        submit,
        [
            "--username",
            "testuser",
            "--namespace",
            "test_namespace",
            "--run-name",
            "test-run",
            "--pipeline",
            "mock_pipeline2",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow=mock_multiple_pipelines,
        pipeline_for_execution="mock_pipeline2",
        verbose=False,
        dry_run=False,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_dry_run(mock_multiple_pipelines: None, mock_submit_internal: None):
    runner = CliRunner()
    result = runner.invoke(
        submit,
        [
            "--username",
            "testuser",
            "--namespace",
            "test_namespace",
            "--run-name",
            "test-run",
            "--pipeline",
            "mock_pipeline2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow=mock_multiple_pipelines,
        pipeline_for_execution="mock_pipeline2",
        verbose=False,
        dry_run=True,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_verbose(mock_multiple_pipelines: None, mock_submit_internal: None):
    runner = CliRunner()
    result = runner.invoke(
        submit,
        [
            "--username",
            "testuser",
            "--namespace",
            "test_namespace",
            "--run-name",
            "test-run",
            "--pipeline",
            "mock_pipeline3",
            "--verbose",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow=mock_multiple_pipelines,
        pipeline_for_execution="mock_pipeline3",
        verbose=True,
        dry_run=False,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_dry_run_and_verbose(mock_multiple_pipelines: None, mock_submit_internal: None) -> None:
    runner = CliRunner()
    result = runner.invoke(
        submit,
        [
            "--username",
            "testuser",
            "--namespace",
            "test_namespace",
            "--run-name",
            "test-run",
            "--pipeline",
            "mock_pipeline2",
            "--dry-run",
            "--verbose",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow=mock_multiple_pipelines,
        pipeline_for_execution="mock_pipeline2",
        verbose=True,
        dry_run=True,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


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
    build_argo_template("test_run", "testuser", "test_namespace", {"test": MagicMock()}, ArgoResourceConfig())
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
    run_name = "test_run"

    result = save_argo_template(argo_template, run_name, temporary_directory)

    assert Path(result).exists()
    assert Path(result).is_file()


def test_save_argo_template_content(temporary_directory: Path) -> None:
    argo_template = "test template content"
    run_name = "test_run"

    file_path = save_argo_template(argo_template, run_name, temporary_directory)

    with open(file_path, "r") as f:
        content = f.read()

    assert content == argo_template


def test_save_argo_template_returns_string(temporary_directory: Path) -> None:
    argo_template = "test template content"
    run_name = "test_run"

    result = save_argo_template(argo_template, run_name, temporary_directory)

    assert isinstance(result, str)


def test_apply_argo_template(mock_run_subprocess: None) -> None:
    apply_argo_template("test_namespace", Path("sample_template.yml"), verbose=True)
    mock_run_subprocess.assert_called_once()


def test_submit_workflow(mock_run_subprocess: None) -> None:
    mock_run_subprocess.return_value.stdout = '{"metadata": {"name": "test-job"}}'
    submit_workflow("test_run", "test_namespace", "__default__", verbose=True)
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
    assert get_run_name(input_name) == expected_name


@patch("matrix.cli_commands.submit.run_subprocess")
def test_get_run_name_from_git(mock_run_subprocess: None) -> None:
    mock_run_subprocess.return_value.stdout = "feature/test-branch"
    assert get_run_name(None).startswith("feature-test-branch")


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

    _submit(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow={
            "test_pipeline": Pipeline(
                nodes=[
                    Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")
                ]
            ),
            "__default__": Pipeline(
                nodes=[
                    Node(func=dummy_func, inputs=["dataset_a", "dataset_b"], outputs="dataset_c", name="simple_node")
                ]
            ),
        },
        pipeline_for_execution=pipeline_for_execution,
        verbose=False,
        dry_run=False,
        template_directory=temporary_directory,
        allow_interactions=False,
    )

    yaml_files = list(temporary_directory.glob("argo_template_test-run_*.yml"))
    assert len(yaml_files) == 1, f"Expected 1 YAML file, found {len(yaml_files)}"

    yaml_file = yaml_files[0]
    assert yaml_file.is_file(), f"Expected {yaml_file} to be a file"
    assert yaml_file.name.startswith("argo_template_test-run_"), f"Unexpected file name format: {yaml_file.name}"
    assert yaml_file.name.endswith(".yml"), f"File does not have .yml extension: {yaml_file.name}"

    # Read and parse the YAML file
    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    # Check if the content is a dictionary
    assert isinstance(content, dict), "Parsed YAML content should be a dictionary"

    # Check for the presence of two pipelines in the templates
    templates = content.get("spec", {}).get("templates", [])
    pipeline_templates = [t for t in templates if "dag" in t]

    assert len(pipeline_templates) == 2, "Expected two pipeline templates (test and cloud)"

    pipeline_names = [t["name"] for t in pipeline_templates]
    assert "test_pipeline" in pipeline_names, "Expected 'test' pipeline to be present"
    assert "__default__" in pipeline_names, "Expected 'cloud' pipeline to be present"

    # Additional checks
    assert content["metadata"]["name"] == "test-run", "Expected 'test-run' as the workflow name"
    assert content["metadata"]["namespace"] == "test_namespace", "Expected 'test_namespace' as the namespace"

    # Check for the presence of tasks in each pipeline
    for pipeline in pipeline_templates:
        tasks = pipeline.get("dag", {}).get("tasks", [])
        assert len(tasks) > 0, f"Expected at least one task in the {pipeline['name']} pipeline"

    # Check that the specified pipeline_for_execution is present in the templates
    assert pipeline_for_execution in pipeline_names, f"Expected '{pipeline_for_execution}' pipeline to be present"
    # NOTE: This function was partially generated using AI assistance.

    submit_cmd = " ".join(
        [
            "argo submit",
            "--name test-run",
            "-n test_namespace",
            "--from wftmpl/test-run",
            "-p run_name=test-run",
            "-l submit-from-ui=false",
            f"--entrypoint {pipeline_for_execution}",
            "-o json",
        ]
    )
    mock_run_subprocess.assert_called_with(submit_cmd, capture_output=True, stream_output=False)
