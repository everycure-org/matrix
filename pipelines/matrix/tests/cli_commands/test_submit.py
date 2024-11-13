import pytest
from unittest.mock import patch, MagicMock
import click
from click.testing import CliRunner
from matrix.cli_commands.submit import (
    submit,
    get_run_name,
)


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


@pytest.mark.skip(reason="Investigate why click is not correctly throwing up exceptions")
def test_pipeline_not_found(mock_multiple_pipelines):
    with pytest.raises(click.ClickException):
        # Given a CLI runner instance
        runner = CliRunner()

        # When invoking with non existing pipeline
        runner.invoke(submit, ["--username", "testuser", "--run-name", "test-run", "--pipeline", "not_exists"])

        # Then not found error is thrown
