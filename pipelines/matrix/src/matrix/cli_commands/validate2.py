from typing import Any, Collection, Dict, List, NamedTuple, Optional, Set

import click
from google.cloud import storage
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import env_option
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.framework.project import pipelines, settings
from kedro.io import DataCatalog
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from matrix.session import KedroSessionWithFromCatalog

console = Console()


class ValidateConfig(NamedTuple):
    env: str
    conf_source: Optional[str]
    from_env: Optional[str]


@project_group.command()
@env_option
@click.option(
    "--conf-source", type=click.Path(exists=True, file_okay=False, resolve_path=True), help="Specify the config source."
)
@click.option("--from-env", type=str, default=None, help="Custom env to read from.")
def validate(env: str, conf_source: Optional[str], from_env: Optional[str] = None):
    """Validate the existence of datasets stored on Google Cloud Storage."""
    console.rule("[bold blue]Validating GCS Datasets")

    config = ValidateConfig(env=env, conf_source=conf_source, from_env=from_env)
    _validate(config, KedroSessionWithFromCatalog)


def _validate(config: ValidateConfig, kedro_session: KedroSessionWithFromCatalog) -> None:
    storage_client = storage.Client()

    with kedro_session.create(env=config.env, conf_source=config.conf_source) as session:
        from_catalog = _extract_config(config, session)
        if from_catalog is None:
            raise RuntimeError("No data catalog found.")

        missing_files = []
        table = Table(title="GCS Dataset Validation", show_header=True, header_style="bold magenta")
        table.add_column("Dataset Name", style="cyan")
        table.add_column("GCS Path", style="yellow")
        table.add_column("Exists", style="green")

        for dataset_name, dataset in from_catalog._data_sets.items():
            if hasattr(dataset, "_filepath") and dataset._filepath.startswith("gs://"):
                gcs_path = dataset._filepath
                bucket_name, blob_path = gcs_path[5:].split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                exists = blob.exists()

                if not exists:
                    missing_files.append(gcs_path)

                table.add_row(dataset_name, gcs_path, "✅" if exists else "❌")

        console.print(table)

        if missing_files:
            console.print("[bold red]Some datasets are missing in GCS![/bold red]")
            for file in missing_files:
                console.print(f"[red]Missing:[/red] {file}")
            exit(1)
        else:
            console.print("[bold green]All GCS datasets are present.[/bold green]")


def _extract_config(config: ValidateConfig, session: KedroSessionWithFromCatalog) -> Optional[DataCatalog]:
    from_catalog: Optional[DataCatalog] = None
    if config.from_env:
        config_loader_class = settings.CONFIG_LOADER_CLASS
        config_loader = config_loader_class(
            conf_source=session._conf_source, env=config.from_env, **settings.CONFIG_LOADER_ARGS
        )
        conf_catalog = config_loader["catalog"]
        conf_catalog = _convert_paths_to_absolute_posix(
            project_path=session._project_path, conf_dictionary=conf_catalog
        )
        conf_creds = config_loader["credentials"]
        from_catalog = settings.DATA_CATALOG_CLASS.from_config(catalog=conf_catalog, credentials=conf_creds)
    return from_catalog
