"""Command line interface for Matrix project."""

import click
from kedro.framework.cli.utils import KedroCliError

from matrix.cli_commands.submit import submit
from matrix.cli_commands.run import run


@click.group()
def cli():
    """Matrix CLI tools."""
    pass


# Add the submit command
cli.add_command(submit)

# Add the run command
cli.add_command(run)

if __name__ == "__main__":
    try:
        cli()
    except KedroCliError as e:
        raise e
    except Exception as e:
        raise KedroCliError(str(e))
