"""CLI for matrix-validator."""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import click

from matrix_validator import __version__, validator_polars, validator_purepython, validator_schema

logger = logging.getLogger("matrix-validator")

app = typer.Typer()


class ValidatorImplementationChoice(str, Enum):
    pandas = "pandas"
    python = "python"
    polars = "polars"


class OutputFormatChoice(str, Enum):
    txt = "txt"
    md = "md"


@app.command()
def main(
    nodes: Annotated[Path, typer.Option("--nodes", "-n", help="Path to the nodes TSV file.")],
    edges: Annotated[Path, typer.Option("--edges", "-e", help="Path to the edges TSV file.")],
    verbose: int = 0,
    quiet: bool = False,
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to the config file.")] = None,
    validator: Annotated[
        ValidatorImplementationChoice, typer.Option("--validator", "-n")
    ] = ValidatorImplementationChoice.polars,
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Rows to validate.  When not set, all rows are validated.")
    ] = None,
    output_format: Annotated[
        OutputFormatChoice, typer.Option("--output-format", "-o", help="Format of the validation report.")
    ] = OutputFormatChoice.txt,
):
    """Run the Matrix Validator CLI."""
    match verbose:
        case 2:
            level = logging.DEBUG
        case 1:
            level = logging.INFO
        case _:
            level = logging.WARNING

    if quiet:
        level = logging.ERROR

    logging.basicConfig(stream=sys.stdout, level=level)

    if quiet:
        logging.basicConfig(stream=sys.stdout, level=logging.FATAL)

    match validator:
        case "python":
            python(config, nodes, edges, limit, output_format)
        case "pandera":
            pandera(config, nodes, edges, limit, output_format)
        case "polars":
            polars(config, nodes, edges, limit, output_format)


def polars(config, nodes, edges, limit, output_format):
    """
    CLI for matrix-validator.

    This validates a knowledge graph using optional nodes and edges TSV files.
    """
    exit_code = 0
    try:
        validator = validator_polars.ValidatorPolarsFileImpl(nodes, edges, config)
        if output_format:
            validator.set_output_format(output_format)
        exit_code = validator.validate(limit)
    except Exception as e:
        logger.exception(f"Error during validation: {e}")
        click.echo("Validation failed. See logs for details.", err=True)
    sys.exit(exit_code)


def python(config, nodes, edges, limit, output_format):
    """
    CLI for matrix-validator.

    This validates a knowledge graph using optional nodes and edges TSV files.
    """
    exit_code = 0
    try:
        validator = validator_purepython.ValidatorPurePythonImpl(config)
        if output_format:
            validator.set_output_format(output_format)
        exit_code = validator.validate(nodes, edges, limit)
    except Exception as e:
        logger.exception(f"Error during validation: {e}")
        click.echo("Validation failed. See logs for details.", err=True)
    sys.exit(exit_code)


def pandera(config, nodes, edges, limit, output_format):
    """
    CLI for matrix-validator.

    This validates a knowledge graph using optional nodes and edges TSV files.
    """
    exit_code = 0
    try:
        validator = validator_schema.ValidatorPanderaImpl(config)
        if output_format:
            validator.set_output_format(output_format)
        exit_code = validator.validate(nodes, edges, limit)
    except Exception as e:
        logger.exception(f"Error during validation: {e}")
        click.echo("Validation failed. See logs for details.", err=True)
    sys.exit(exit_code)


def main() -> None:
    """Wrap the Robokop CLI options in a Typer app."""
    app()


if __name__ == "__main__":
    main()
