from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from matrix.cli_commands.submit import (
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
from kedro.framework.project import pipelines as kedro_pipelines


@pytest.fixture
def mock_run_subprocess():
    with patch("matrix.cli_commands.submit.run_subprocess") as mock:
        yield mock


@pytest.fixture
def mock_dependencies():
    with patch("matrix.cli_commands.submit.check_dependencies") as _, patch(
        "matrix.cli_commands.submit.build_push_docker"
    ) as _, patch("matrix.cli_commands.submit.build_argo_template") as _, patch(
        "matrix.cli_commands.submit.ensure_namespace"
    ) as _, patch("matrix.cli_commands.submit.apply_argo_template") as _, patch(
        "matrix.cli_commands.submit.submit_workflow"
    ) as _, patch("matrix.cli_commands.submit.get_run_name", return_value="test-run"):
        yield


@pytest.fixture
def mock_submit_internal():
    with patch("matrix.cli_commands.submit._submit") as mock:
        yield mock


@pytest.fixture
def mock_submit_internal():
    with patch("matrix.cli_commands.submit._submit") as mock:
        yield mock


@pytest.fixture
def mock_pipelines():
    pipelines = {
        "mock_pipeline": MagicMock(),
    }

    with patch("matrix.cli_commands.submit.kedro_pipelines", return_value=pipelines) as mock:
        yield mock


@pytest.fixture
def mock_multiple_pipelines():
    pipelines = {
        "mock_pipeline": MagicMock(),
        "mock_pipeline2": MagicMock(),
        "mock_pipeline3": MagicMock(),
    }

    with patch("matrix.cli_commands.submit.kedro_pipelines", return_value=pipelines) as mock:
        yield mock


def test_submit_simple(mock_submit_internal: None, mock_pipelines: None) -> None:
    runner = CliRunner()
    result = runner.invoke(submit, ["--username", "testuser", "--run-name", "test-run"])
    assert result.exit_code == 0

    mock_submit_internal.assert_called_once_with(
        username="testuser",
        namespace="argo-workflows",
        run_name="test-run",
        pipelines=mock_pipelines,
        verbose=False,
        dry_run=False,
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
        pipelines=mock_pipelines,
        verbose=False,
        dry_run=False,
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
        pipelines={"mock_pipeline2": mock_multiple_pipelines["mock_pipeline2"]},
        verbose=False,
        dry_run=False,
    )


@pytest.mark.skip(reason="Test broken, needs to be fixed")
def test_submit_multiple_pipelines(mock_submit_internal: None):
    runner = CliRunner()
    result = runner.invoke(
        submit,
        [
            "--username",
            "testuser",
            "--namespace",
            "test_namespace",
            "--pipeline",
            "mock_pipeline2",
            "--pipeline",
            "mock_pipeline3",
        ],
    )
    assert result.exit_code == 0
    mock_submit_internal.assert_called_once_with(
        "testuser", "test_namespace", "test-run", ("test_pipeline", "test_pipeline2"), False, False
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
