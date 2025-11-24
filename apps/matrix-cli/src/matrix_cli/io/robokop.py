"""
Python port of https://github.com/jdr0887/rusty-matrix-io/blob/main/src/bin/robokop.rs.

Subcommands:
- build-edges: Transform a Robokop edges TSV into KGX-like edges with Biolink predicates.
- build-nodes: Join multiple node feature TSVs and map types/IDs to CURIEs.
- print-predicate-mappings: Infer mapping table between relation/display_relation and Biolink predicate; print as pretty JSON to stdout.

Examples:
  primekg build-edges -i kg.tsv -o edges.tsv
  primekg build-nodes -a drug_features.tsv -b disease_features.tsv -n nodes.tsv -o nodes_out.tsv
  primekg print-predicate-mappings -i primekg.tsv

"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from matrix_io_utils.robokop import (
    robokop_convert_boolean_columns_to_label_columns,
    robokop_strip_type_from_column_names,
)

from matrix_cli.io import read_tsv_lazy

app = typer.Typer()


@app.command()
def convert_boolean_columns_to_label_columns(
    nodes: Annotated[Path, typer.Option("--nodes", "-n")],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Build the nodes tsv file."""
    nodes_df = read_tsv_lazy(nodes, delimiter="\t")
    nodes_df = robokop_convert_boolean_columns_to_label_columns(nodes_df)
    output.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.write_csv(output, separator="\t")


@app.command()
def strip_type_from_column_names(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Build the nodes tsv file."""
    input_df = read_tsv_lazy(input, delimiter="\t")
    input_df = robokop_strip_type_from_column_names(input_df)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_df.write_csv(output, separator="\t")


def main() -> None:
    """Wrap the Robokop CLI options in a Typer app."""
    app()


if __name__ == "__main__":
    main()
