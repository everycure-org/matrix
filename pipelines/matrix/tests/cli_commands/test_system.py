import subprocess
from unittest.mock import MagicMock, patch

import pytest
from matrix_auth.system import command_exists, run_subprocess


@pytest.fixture
def mock_run_subprocess():
    with patch("matrix_auth.system.run_subprocess") as mock:
        mock.return_value = MagicMock(stdout='{"metadata": {"name": "mocked-job-name"}}')
        yield mock


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


@pytest.mark.parametrize("stream", ("/dev/stdout", "/dev/stderr"))
@pytest.mark.timeout(15)
def test_run_subprocess_no_deadlock(stream: str) -> None:
    """Reproduces an annoying deadlocking issue with the way run_subprocess in streaming mode was written up to c0f2f3f."""
    # Put "lots" of output in one of stdout or stderr.
    big_number = 100_000  # big enough to fill a process pipe, though that is platform dependant
    cmd = f"yes | head -n {big_number} > {stream}"
    finished_process = run_subprocess(cmd, stream_output=True, check=True, shell=True)
    channel = finished_process.stdout if stream == "/dev/stdout" else finished_process.stderr
    assert len(channel) == big_number * len("y\n")
