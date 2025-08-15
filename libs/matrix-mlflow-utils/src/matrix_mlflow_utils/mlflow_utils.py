import os
import secrets
from datetime import datetime, timezone
from typing import Optional

import mlflow
from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient
from rich.console import Console

console = Console()


class ExperimentNotFound(Exception):
    "Raised when an no experiment is found with provided name"

    pass


class DeletedExperimentExistsWithName(Exception):
    "Raised when an experiment exists with the desired name, but has been soft-deleted"

    pass


EXPERIMENT_ARCHIVE_EXCLUSION_LIST = [
    "archive",
    "test-caching",
    "run_node2vec_iter10",
    "full_matrix_run_10",
    "feature-attribute-embeddings-strategy-93d6bec0",
    "feature-attribute-embeddings-strategy-4c53361f",
    "feature-attribute-embeddings-strategy-94019e37",
    "lc-baseline-run-run-23-aug-setup-3_FINAL",
    "fix-mlflow-modelling-bug",
    "sample-run",
    "Default",
]
ARCHIVE_EXPERIMENT_ID = 17365
# Choose a cutoff date for experiments to archive
# Note that MLflow uses millisecond timestamps
CUTOFF_TIMESTAMP = int(datetime(2025, 2, 21, tzinfo=timezone.utc).timestamp() * 1000)


def create_mlflow_experiment(experiment_name: str) -> str:
    """Creates an MLflow experiment with a given name.

    If an experiment with the same name exists but is deleted, the user is prompted
    to rename the deleted experiment to allow creation of a new one.
    """

    existing_exp = mlflow.get_experiment_by_name(name=experiment_name)

    if existing_exp:
        if existing_exp.lifecycle_stage == "deleted":
            console.print(f"Error: Experiment '{experiment_name}' exists but is deleted.", existing_exp.experiment_id)
            raise DeletedExperimentExistsWithName()
        else:
            raise Exception(
                f"❌ Error: Experiment '{experiment_name}' already exists at {mlflow.get_tracking_uri()}/#/experiments/{existing_exp.experiment_id}"
            )

    artifact_location = f"gs://{os.environ['RUNTIME_GCP_BUCKET']}/kedro/mlflow/{experiment_name}"
    console.print(f"Creating new experiment: '{experiment_name}'")
    experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
    console.print(f"✅ Experiment '{experiment_name}' created with ID {experiment_id}.")

    return experiment_id


def get_experiment_id_from_name(experiment_name: str) -> Optional[str]:
    """Fetches the experiment ID from MLflow by experiment name."""
    experiment = mlflow.get_experiment_by_name(name=experiment_name)

    if experiment and experiment.lifecycle_stage == "active":
        console.print(
            f"✅ Experiment '{experiment_name}' found at {mlflow.get_tracking_uri()}/#/experiments/{experiment.experiment_id}"
        )
        return experiment.experiment_id
    else:
        raise ExperimentNotFound(
            f"❌ Experiment '{experiment_name}' not found. Please create an experiment using `kedro experiment create` before proceeding."
        )


def rename_soft_deleted_experiment(experiment_name: str) -> str:
    """
    Restore, rename and re-delete the deleted experiment

    Args:
         experiment_name: name of the deleted experiment

    Returns:
        str: the new name of the deleted experiment
    """
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    experiment_id = str(experiment.experiment_id)

    try:
        client.restore_experiment(experiment_id)
        new_deleted_name = f"deleted-{experiment.name}-{secrets.token_hex(4)}"
        client.rename_experiment(experiment_id, new_deleted_name)
        client.delete_experiment(experiment_id)

        console.print(f"Renamed deleted experiment to '{new_deleted_name}' and removed it.")

        return new_deleted_name

    except Exception as e:
        console.print(f"Error renaming deleted experiment: {e}")
        raise e


def copy_run(original_run: Run, new_experiment_id: int):
    console.print(f"Copying run '{original_run.info.run_name}' to experiment {new_experiment_id}")
    client = mlflow.tracking.MlflowClient()

    new_run = client.create_run(
        experiment_id=new_experiment_id,
        tags=original_run.data.tags,
        run_name=original_run.info.run_name,
    )

    # Update run status to match original
    client.update_run(new_run.info.run_id, status=original_run.info.status)
    #   Copy metrics and params
    for metric in original_run.data.metrics.items():
        client.log_metric(new_run.info.run_id, metric[0], metric[1])
    for param in original_run.data.params.items():
        client.log_param(new_run.info.run_id, param[0], param[1])

    # Copy dataset inputs if they exist
    if hasattr(original_run, "inputs") and original_run.inputs.dataset_inputs:
        client.log_inputs(
            run_id=new_run.info.run_id,
            datasets=original_run.inputs.dataset_inputs,
        )

    console.print(
        f"Run {original_run.info.run_name} ({original_run.info.run_id}) copied to experiment {new_experiment_id}"
    )


def delete_run(client: MlflowClient, run_id: str):
    try:
        run_to_delete = client.get_run(run_id)
    except Exception as e:
        console.print(f"Run {run_id} not found")
        raise e

    try:
        client.delete_run(run_to_delete.info.run_id)
    except Exception as e:
        console.print(f"Error deleting run {run_id}: {e}")
        raise e


def archive_runs_and_experiments(dry_run: bool = True):
    """Archives MLflow runs and experiments by copying them to an archive experiment.

    Searches MLFlow for all experiments and:
    1. Skips experiments that are in the EXPERIMENT_ARCHIVE_EXCLUSION_LIST (for example, the archive experiment itself, and any other experiments that we do not want to archive)
    2. For each remaining experiment:
        - Finds all active runs
            - Copies each run to the archive experiment
            - Soft deletes the original runs
        - Soft deletes the original experiment
    """
    client = MlflowClient()

    experiments = client.search_experiments()
    console.print(f"Found {len(experiments)} experiments")

    for experiment in experiments:
        if experiment.creation_time > CUTOFF_TIMESTAMP:
            console.print(f"Skipping experiment: {experiment.name} (creation time after cutoff)")
            continue

        if experiment.name in EXPERIMENT_ARCHIVE_EXCLUSION_LIST:
            console.print(f"Skipping experiment: {experiment.name} (from exclusion list)")
            continue

        runs = client.search_runs(experiment_ids=[experiment.experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
        console.print(f"Found {len(runs)} runs in experiment {experiment.name}")

        for run in runs:
            if dry_run:
                print(
                    f"Dry run. Would archive run {run.info.run_name} ({run.info.run_id}) to experiment {ARCHIVE_EXPERIMENT_ID}"
                )
            else:
                copy_run(original_run=run, new_experiment_id=ARCHIVE_EXPERIMENT_ID)
                delete_run(client, run.info.run_id)
                console.print(f"Original run {run.info.run_name} ({run.info.run_id}) deleted")
        try:
            if dry_run:
                print(f"Dry run. Would delete experiment {experiment.name}")
            else:
                client.delete_experiment(experiment.experiment_id)
                console.print(f"Experiment {experiment.name} deleted")
        except Exception as e:
            console.print(f"Error deleting experiment {experiment.name}: {e}")
            raise e
