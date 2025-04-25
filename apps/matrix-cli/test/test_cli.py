# smoketest that checks the CLI is set up correctly
from typing import List, Union

import pytest
from typer.testing import CliRunner

from matrix_cli.main import app

runner = CliRunner()


@pytest.mark.parametrize(
    "subcommand", ["code", "data", ["data", "download"], "gh-users", "releases"]
)
def test_app(subcommand: Union[str, List[str]]):
    command_works(subcommand)


def command_works(command: Union[str, List[str]]):
    if isinstance(command, str):
        command = [command]
    result = runner.invoke(app, command + ["--help"])
    assert result.exit_code == 0
    assert f"Usage: root {' '.join(command)}" in result.stdout
