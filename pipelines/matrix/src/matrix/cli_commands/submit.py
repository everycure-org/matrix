import logging

import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


console = Console()


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@cli.command()
def submit():
    """Submit the end-to-end workflow."""
    click.secho("kedro submit has been deprecated. Please use `kedro experiment run`.", bg="red", fg="black")
    click.Abort()
