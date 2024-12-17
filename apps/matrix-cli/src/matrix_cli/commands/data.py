import os
import subprocess
from pathlib import Path
from typing import Dict

import typer

from matrix_cli.components.settings import settings
from matrix_cli.components.utils import console

data_app = typer.Typer(
    help="Data-related utility commands, inspired by the kaggle dataset download CLI", no_args_is_help=True
)


@data_app.command()
def download(
    data_type: str = typer.Argument(default="raw", help="Data type to pull"),
    target_dir: str = typer.Argument(..., help="Target directory to pull raw data to"),
    dry_run: bool = typer.Option(False, help="Dry run the synchronization"),
):
    """Pull raw data down to the local machine."""
    # prep paths
    data_paths = _get_data_path(data_type, target_dir)

    _check_if_in_root_else_give_warning()

    try:
        for gcs_uri, local_dir in data_paths.items():
            os.makedirs(local_dir, exist_ok=True)
            sync_gcs_to_local(gcs_uri, local_dir, dry_run)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise typer.Exit(1)


def _check_if_in_root_else_give_warning():
    if not (Path.cwd() / "conf").exists():
        console.print(
            "[bold yellow]Warning: You are not in the pipeline root directory. This command is intended to be run from the pipeline root directory. Are you sure you want to continue?"
        )
        typer.confirm("Continue?", abort=True)


def _get_data_path(data_type: str, target_dir: str) -> Dict[str, Path]:
    # currently only support raw
    if data_type == "raw":
        paths = settings.raw_paths
        return {f"{settings.gcs_base_uri}/{path}": Path(target_dir) / path for path in paths}

    # finally
    raise ValueError(f"Unsupported data type: {data_type}")


def sync_gcs_to_local(gcs_uri, local_dir, dry_run=False):
    command = ["gsutil", "-m", "rsync", "-r", gcs_uri, local_dir]
    if dry_run:
        console.print(f"[bold yellow]Dry run: \n{' '.join(map(str, command))}")
    else:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[bold green]Successfully synchronized {gcs_uri} to {local_dir}")
        else:
            console.print(f"[bold red]Error: {result.stderr}")
