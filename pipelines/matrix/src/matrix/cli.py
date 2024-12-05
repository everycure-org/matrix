import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS, KedroCliError

from matrix.cli_commands.extract import extract
from matrix.cli_commands.run import run
from matrix.cli_commands.submit import submit


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    pass


cli.add_command(submit)

# Add the run command
cli.add_command(run)
# Add the extract command
cli.add_command(extract)

if __name__ == "__main__":
    try:
        cli()
    except KedroCliError as e:
        raise e
    except Exception as e:
        raise KedroCliError(str(e))
