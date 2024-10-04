import click
from kedro.framework.cli.utils import KedroCliError

from matrix.cli_commands.submit import submit
from matrix.cli_commands.run import run

from kedro.framework.cli.utils import CONTEXT_SETTINGS


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    pass


cli.add_command(submit)

cli.add_command(run)

if __name__ == "__main__":
    try:
        cli()
    except KedroCliError as e:
        raise e
    except Exception as e:
        raise KedroCliError(str(e))
