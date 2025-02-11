import os
import secrets
from typing import List, Optional

import mlflow
from rich.console import Console

console = Console()


class ExperimentNotFound(Exception):
    "Raised when an no experiment is found with provided name"

    pass


class DeletedExperimentExistsWithName(Exception):
    "Raised when an experiment exists with the desired name, but has been soft-deleted"

    pass


def create_mlflow_experiment(experiment_name: str) -> str:
    """Creates an MLflow experiment with a given name.

    If an experiment with the same name exists but is deleted, the user is prompted
    to rename the deleted experiment to allow creation of a new one.
    """
    existing_exp = mlflow.get_experiment_by_name(name=experiment_name)

    if existing_exp:
        if existing_exp.lifecycle_stage == "deleted":
            raise DeletedExperimentExistsWithName(
                f"Error: Experiment '{experiment_name}' exists but is deleted.", existing_exp.experiment_id
            )
        else:
            raise Exception(
                f"❌ Error: Experiment '{experiment_name}' already exists at {mlflow.get_tracking_uri()}/#/experiments/{existing_exp.experiment_id}"
            )

    artifact_location = f"gs://{os.environ['GCP_BUCKET']}/kedro/mlflow/{experiment_name}"
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
