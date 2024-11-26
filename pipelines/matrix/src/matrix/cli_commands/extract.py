# NOTE: This file was partially generated using AI assistance.

from pathlib import Path

import click
import yaml
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.io.core import DatasetNotFoundError


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@cli.command()
@click.option("--pipeline", "-p", default="__default__", required=True, help="Name of the pipeline to analyze")
@click.option("--env", "-e", default="cloud", required=True, help="Environment name (e.g. local, cloud)")
@click.option(
    "--format", "-f", default="json", type=click.Choice(["yaml", "json"]), help="Output format (yaml or json)"
)
@click.option("--output-file", "-o", type=click.Path(), help="Output file path (if not specified, prints to stdout)")
def extract(pipeline: str, env: str, format: str, output_file: str):
    """Analyze all outputs from a pipeline and their catalog destinations."""

    # Get the pipeline object
    if pipeline not in pipelines:
        raise click.BadParameter(f"Pipeline '{pipeline}' not found. Available pipelines: {list(pipelines.keys())}")

    pipeline_obj = pipelines[pipeline]

    # Create session with specified environment
    with KedroSession.create(env=env) as session:
        context = session.load_context()
        catalog = context.catalog

        # Collect all outputs and their paths
        output_mapping = {}

        # Get all outputs from the pipeline
        for node in pipeline_obj.nodes:
            for output in node.outputs:
                try:
                    # Try to get the dataset from the catalog
                    dataset = catalog._get_dataset(output)

                    if hasattr(dataset, "_get_load_path"):
                        load_path = dataset._get_load_path()
                    else:
                        load_path = None

                    mapping = {
                        "load_path": str(load_path),
                        "type": dataset._dataset_type if hasattr(dataset, "_dataset_type") else type(dataset).__name__,
                        "details": dataset._describe(),
                    }
                    if mapping["details"].get("filepath", None):
                        mapping["details"]["filepath"] = str(mapping["details"]["filepath"])
                    output_mapping[output] = mapping

                except DatasetNotFoundError:
                    # Handle dynamic catalog entries that haven't been resolved
                    click.echo(
                        f"Warning: Dataset '{output}' is not directly found in catalog (might be a template)", err=True
                    )
                    output_mapping[output] = {
                        "filepath": "DYNAMIC_ENTRY - needs to be resolved",
                        "type": "Unknown - dynamic entry",
                        "format": "Unknown - dynamic entry",
                    }

        # Format output based on chosen format
        if format == "yaml":
            formatted_output = yaml.dump(output_mapping, sort_keys=False, default_flow_style=False)
        else:  # json
            import json

            formatted_output = json.dumps(output_mapping, indent=2)

        # Output to file or stdout
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(formatted_output)
            click.echo(f"Output written to {output_path}")
        else:
            click.echo(formatted_output)
