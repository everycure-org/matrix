import secrets
from typing import List, Optional

import click
import mlflow


def create_mlflow_experiment(experiment_name: str) -> str:
    """Creates an MLflow experiment with a given name.

    If an experiment with the same name exists but is deleted, the user is prompted
    to rename the deleted experiment to allow creation of a new one.
    """
    client = mlflow.tracking.MlflowClient()
    existing_exp = mlflow.get_experiment_by_name(name=experiment_name)

    if existing_exp:
        if existing_exp.lifecycle_stage == "deleted":
            click.echo(f"Error: Experiment '{experiment_name}' exists but is deleted.")

            if click.confirm(
                f"Would you like to rename the deleted experiment and continue using the name '{experiment_name}'?",
                abort=True,
            ):
                try:
                    # Restore, rename and re-delete the deleted experiment
                    client.restore_experiment(existing_exp.experiment_id)
                    new_deleted_name = f"deleted-{experiment_name}-{secrets.token_hex(4)}"
                    client.rename_experiment(existing_exp.experiment_id, new_deleted_name)
                    client.delete_experiment(existing_exp.experiment_id)

                    click.echo(f"Renamed deleted experiment to '{new_deleted_name}' and removed it.")

                    # Create a new experiment
                    experiment_id = mlflow.create_experiment(name=experiment_name)
                    click.echo(f"✅ New experiment '{experiment_name}' created with ID {experiment_id}.")
                    return experiment_id

                except Exception as e:
                    click.echo(f"An error occurred: {e}")
                    raise click.Abort()

        else:
            click.echo(
                f"❌ Error: Experiment '{experiment_name}' already exists at {mlflow.get_tracking_uri()}/#/experiments/{existing_exp.experiment_id}"
            )
            raise click.Abort()

    # If no existing experiment, create a new one
    click.echo(f"Creating new experiment: '{experiment_name}'")
    experiment_id = mlflow.create_experiment(name=experiment_name)
    click.echo(f"✅ Experiment '{experiment_name}' created with ID {experiment_id}.")

    return experiment_id


def get_experiment_id_from_name(experiment_name: str) -> Optional[str]:
    """Fetches the experiment ID from MLflow by experiment name."""
    experiment = mlflow.get_experiment_by_name(name=experiment_name)

    if experiment:
        click.echo(
            f"✅ Experiment '{experiment_name}' found at {mlflow.get_tracking_uri()}/#/experiments/{experiment.experiment_id}"
        )
        return experiment.experiment_id
    else:
        click.echo(f"❌ Experiment '{experiment_name}' not found.")
        click.echo("ℹ️ Please create an experiment using `kedro experiment create` before proceeding.")
        raise click.Abort()
