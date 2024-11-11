import os
import subprocess
from pathlib import Path

import typer

from matrix_cli.settings import settings
from matrix_cli.utils import console

data_app = typer.Typer(help="Data-related utility commands", no_args_is_help=True)
raw_app = typer.Typer(help="Raw data-related utility commands", no_args_is_help=True)

data_app.add_typer(
    raw_app,
    name="raw",
    help="Raw data-related utility commands",
)


@raw_app.command()
def pull(
    target_dir: str = typer.Argument(..., help="Target directory to pull raw data to"),
    dry_run: bool = typer.Option(False, help="Dry run the synchronization"),
):
    """Pull raw data down to the local machine."""
    # prep paths
    raw_path = Path(target_dir) / "data/01_RAW"
    kedro_raw_path = Path(target_dir) / "kedro/data/01_raw"

    try:
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(kedro_raw_path, exist_ok=True)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise typer.Exit(1)

    gs_raw_uri = f"{settings.gcs_base_uri}/data/01_RAW"
    gs_kedro_raw_uri = f"{settings.gcs_base_uri}/kedro/data/01_raw"

    # trigger downloads
    console.print(f"[bold green]Synchronizing raw data from {gs_raw_uri} to {raw_path}")
    sync_gcs_to_local(gs_raw_uri, raw_path, dry_run)
    console.print(f"[bold green]Synchronizing raw data from {gs_kedro_raw_uri} to {kedro_raw_path}")
    sync_gcs_to_local(gs_kedro_raw_uri, kedro_raw_path, dry_run)


def sync_gcs_to_local(gcs_uri, local_dir, dry_run=False):
    command = ["gsutil", "-m", "rsync", "-r", gcs_uri, local_dir]
    if dry_run:
        console.print(f"[bold yellow]Dry run: \n{' '.join([str(x) for x in command])}")
    else:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[bold green]Successfully synchronized {gcs_uri} to {local_dir}")
        else:
            console.print(f"[bold red]Error: {result.stderr}")
