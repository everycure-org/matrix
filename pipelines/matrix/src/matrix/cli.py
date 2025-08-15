import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS, KedroCliError

from matrix.cli_commands.experiment import experiment
from matrix.cli_commands.run import run


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    pass


logging.info("registering custom matrix kedro CLI extensions")
cli.add_command(run)
cli.add_command(experiment)

if __name__ == "__main__":
    try:
        cli()
    except KedroCliError as e:
        raise e
    except Exception as e:
        raise KedroCliError(str(e))
