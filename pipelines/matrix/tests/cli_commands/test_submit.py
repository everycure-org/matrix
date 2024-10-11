from pathlib import Path
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
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


@pytest.fixture
def mock_run_subprocess():
    with patch("matrix.cli_commands.submit.run_subprocess") as mock:
        yield mock


@pytest.fixture
def mock_dependencies():
    with patch("matrix.cli_commands.submit.check_dependencies") as _, patch(
        "matrix.cli_commands.submit.build_push_docker"
    ) as _, patch("matrix.cli_commands.submit.apply_argo_template") as _, patch(
        "matrix.cli_commands.submit.ensure_namespace"
    ) as _, patch("matrix.cli_commands.submit.submit_workflow"):
        yield

    # build save and apply are only ones left


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
            "--include_pipeline",
            "mock_pipeline2",
            "--pipeline_for_execution",
            "mock_pipeline2",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow={"mock_pipeline2": mock_multiple_pipelines["mock_pipeline2"]},
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
            "--include_pipeline",
            "mock_pipeline2",
            "--include_pipeline",
            "mock_pipeline3",
            "--pipeline_for_execution",
            "mock_pipeline2",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow={
            "mock_pipeline2": mock_multiple_pipelines["mock_pipeline2"],
            "mock_pipeline3": mock_multiple_pipelines["mock_pipeline3"],
        },
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
            "--include_pipeline",
            "mock_pipeline2",
            "--pipeline_for_execution",
            "mock_pipeline2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow={"mock_pipeline2": mock_multiple_pipelines["mock_pipeline2"]},
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
            "--include_pipeline",
            "mock_pipeline3",
            "--pipeline_for_execution",
            "mock_pipeline3",
            "--verbose",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow={"mock_pipeline3": mock_multiple_pipelines["mock_pipeline3"]},
        pipeline_for_execution="mock_pipeline3",
        verbose=True,
        dry_run=False,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_submit_dry_run_and_verbose(mock_multiple_pipelines: None, mock_submit_internal: None):
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
            "--include_pipeline",
            "mock_pipeline2",
            "--include_pipeline",
            "mock_pipeline3",
            "--pipeline_for_execution",
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
        pipelines_for_workflow={
            "mock_pipeline2": mock_multiple_pipelines["mock_pipeline2"],
            "mock_pipeline3": mock_multiple_pipelines["mock_pipeline3"],
        },
        pipeline_for_execution="mock_pipeline2",
        verbose=True,
        dry_run=True,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def test_check_dependencies(mock_run_subprocess):
    mock_run_subprocess.return_value.returncode = 0
    mock_run_subprocess.return_value.stdout = "active_account"
    check_dependencies(verbose=True)
    assert mock_run_subprocess.call_count > 0


def test_build_push_docker(mock_run_subprocess):
    build_push_docker("testuser", verbose=True)
    mock_run_subprocess.assert_called_once_with("make docker_push TAG=testuser", stream_output=True)


@patch("matrix.cli_commands.submit.generate_argo_config")
def test_build_argo_template(mock_generate_argo_config):
    build_argo_template("test_run", "testuser", "test_namespace", {"test": MagicMock()})
    mock_generate_argo_config.assert_called_once()


def test_ensure_namespace_existing(mock_run_subprocess):
    mock_run_subprocess.return_value.returncode = 0
    ensure_namespace("existing_namespace", verbose=True)
    assert mock_run_subprocess.call_count == 1


def test_ensure_namespace_new(mock_run_subprocess):
    mock_run_subprocess.side_effect = [MagicMock(returncode=1), MagicMock(returncode=0)]
    ensure_namespace("new_namespace", verbose=True)
    assert mock_run_subprocess.call_count == 2


@pytest.fixture()
def mock_template_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_save_argo_template_creates_file(mock_template_directory):
    argo_template = "test template content"
    run_name = "test_run"

    result = save_argo_template(argo_template, run_name, mock_template_directory)

    assert Path(result).exists()
    assert Path(result).is_file()


def test_save_argo_template_content(mock_template_directory):
    argo_template = "test template content"
    run_name = "test_run"

    file_path = save_argo_template(argo_template, run_name, mock_template_directory)

    with open(file_path, "r") as f:
        content = f.read()

    assert content == argo_template


def test_save_argo_template_filename_format(mock_template_directory):
    argo_template = "test template content"
    run_name = "test_run"

    file_path = save_argo_template(argo_template, run_name, mock_template_directory)
    filename = Path(file_path).name

    assert filename.startswith(f"argo_template_{run_name}_")
    assert filename.endswith(".yml")

    # Check if the timestamp in the filename is close to the current time
    timestamp_str = filename.split("_")[-1].split(".")[0]
    file_timestamp = time.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    current_time = time.localtime()

    assert abs(time.mktime(file_timestamp) - time.mktime(current_time)) < 5  # Allow 5 seconds difference


def test_save_argo_template_returns_string(mock_template_directory):
    argo_template = "test template content"
    run_name = "test_run"

    result = save_argo_template(argo_template, run_name, mock_template_directory)

    assert isinstance(result, str)


def test_save_argo_template_creates_directory_if_not_exists(tmp_path):
    non_existent_dir = tmp_path / "non_existent"
    argo_template = "test template content"
    run_name = "test_run"

    result = save_argo_template(argo_template, run_name, non_existent_dir)

    assert non_existent_dir.exists()
    assert non_existent_dir.is_dir()
    assert Path(result).exists()
    assert Path(result).is_file()


def test_apply_argo_template(mock_run_subprocess):
    apply_argo_template("test_namespace", Path("sample_template.yml"), verbose=True)
    mock_run_subprocess.assert_called_once()


def test_submit_workflow(mock_run_subprocess):
    mock_run_subprocess.return_value.stdout = '{"metadata": {"name": "test-job"}}'
    submit_workflow("test_run", "test_namespace", verbose=True)
    assert mock_run_subprocess.call_count == 1


def test_get_run_name_with_input():
    assert get_run_name("custom_name") == "custom_name"


@patch("matrix.cli_commands.submit.run_subprocess")
def test_get_run_name_from_git(mock_run_subprocess):
    mock_run_subprocess.return_value.stdout = "feature/test-branch"
    assert get_run_name(None).startswith("feature-test-branch")


def test_command_exists(mock_run_subprocess):
    mock_run_subprocess.return_value.returncode = 0
    assert command_exists("existing_command") is True

    mock_run_subprocess.return_value.returncode = 1
    assert command_exists("non_existing_command") is False


@pytest.fixture
def mock_popen():
    with patch("subprocess.Popen") as mock:
        yield mock


def test_run_subprocess_success(mock_popen):
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


def test_run_subprocess_error(mock_popen):
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


def test_run_subprocess_no_streaming(mock_popen):
    # Mock subprocess.run for non-streaming case
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args='echo "test"', returncode=0, stdout="output", stderr=""
        )

        result = run_subprocess('echo "test"', stream_output=False)

        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""


def test_run_subprocess_no_streaming_error(mock_popen):
    # Mock subprocess.run for non-streaming case with error
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test", output="", stderr="error")

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            run_subprocess("invalid_command", stream_output=False)

        assert exc_info.value.returncode == 1
        assert "error" in exc_info.value.stderr


def test_internal_submit(mock_dependencies: None):
    _submit(
        username="testuser",
        namespace="test_namespace",
        run_name="test-run",
        pipelines_for_workflow={"test_pipeline": MagicMock()},
        pipeline_for_execution="__default__",
        verbose=False,
        dry_run=True,
    )
