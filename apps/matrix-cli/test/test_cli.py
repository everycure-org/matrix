# smoketest that checks the CLI is set up correctly
from typing import List, Union

from matrix_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_app():
    command_works("code")
    command_works("data")
    command_works(["data", "download"])
    command_works("gh-users")
    command_works("releases")


def command_works(command: Union[str, List[str]]):
    if isinstance(command, str):
        command = [command]
    result = runner.invoke(app, command + ["--help"])
    assert result.exit_code == 0
    assert f"Usage: root {' '.join(command)}" in result.stdout
