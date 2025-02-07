import os
import secrets
from typing import List, Optional

import click
import mlflow
from kedro.framework.cli.utils import split_string

from matrix.cli_commands.submit import submit
from matrix.git_utils import create_new_branch, get_current_git_branch
from matrix.utils.authentication import get_iap_token


@click.group()
def experiment():
    token = get_iap_token()
    mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token
    pass


@experiment.command()
@click.option(
    "--experiment-name",
    type=str,
    help="Optional: specify the experiment name to use. Otherwise, default to branch name",
)
def create(experiment_name):
    if not experiment_name:
        current_branch = get_current_git_branch()

        if current_branch.startswith("experiment/"):
            click.confirm(
                f"Creating new mlflow experiment from current branch {current_branch}, is that correct?", abort=True
            )
            experiment_name = current_branch
        elif current_branch == "main":
            user_input_name = click.prompt("Please enter a name for your experiment", type=str)
            if not user_input_name.startswith("experiment/"):
                experiment_name = f"experiment/{user_input_name}"
            click.echo(f"Checking out new branch: {experiment_name}")
            create_new_branch(experiment_name)
        else:
            click.echo("❌ Error: please begin with an `experiment/` prefixed branch or begin from main")
            raise click.Abort()

    mlflow_id = create_mlflow_experiment(experiment_name=experiment_name)

    click.echo(f"kedro experiment created on branch {experiment_name}")
    click.echo(f"mlflow experiment created: {mlflow.get_tracking_uri()}/#/experiments/{mlflow_id}")


@experiment.command()
# @env_option
# These are all copied directly from submit. If we want to maintain kedro submit functionality I think we need to
# keep the duplication for now. Then we can just rename submit to run and add the extra mlflow steps.
@click.option("--username", type=str, required=True, help="Specify the username to use")
@click.option("--namespace", type=str, default="argo-workflows", help="Specify a custom namespace")
@click.option("--run-name", type=str, default=None, help="Specify a custom run name, defaults to branch")
@click.option("--release-version", type=str, required=True, help="Specify a custom release name")
@click.option("--pipeline", "-p", type=str, default="modelling_run", help="Specify which pipeline to execute")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Disable verbose output")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
@click.option("--from-nodes", type=str, default="", help="Specify nodes to run from", callback=split_string)
@click.option("--is-test", is_flag=True, default=False, help="Submit to test folder")
@click.option("--headless", is_flag=True, default=False, help="Skip confirmation prompt")
@click.option("--environment", "-e", type=str, default="cloud", help="Kedro environment to execute in")
@click.pass_context
def run(
    ctx,
    username: str,
    namespace: str,
    run_name: str,
    release_version: str,
    pipeline: str,
    quiet: bool,
    dry_run: bool,
    from_nodes: List[str],
    is_test: bool,
    headless: bool,
    environment: str,
):
    """Run an experiment."""

    experiment_name = get_current_git_branch()
    if not experiment_name.startswith("experiment/"):
        click.echo(f"Error: current branch does not begin with experiment/")
        raise click.Abort()

    click.confirm(f"Start a new run on experiment {experiment_name}, is that correct?", abort=True)
    experiment_id = get_experiment_id_from_name(experiment_name=experiment_name)

    if not run_name:
        # Note, it is not possible to search mlflow runs by name, so we cannot enforce uniqueness, this is down to the user.
        run_name = click.prompt("Please define a name for your run")

    # Invokes the the submit command and forwards all arguments from current command
    # Temporary measure until we formally deprecate kedro submit
    ctx.forward(submit, experiment_id=experiment_id, run_name=run_name)


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
                    click.echo(f"New experiment '{experiment_name}' created with ID {experiment_id}.")
                    return experiment_id

                except Exception as e:
                    click.echo(f"An error occurred: {e}")
                    raise click.Abort()

        else:
            click.echo(
                f"Error: Experiment '{experiment_name}' already exists at {mlflow.get_tracking_uri()}/#/experiments/{existing_exp.experiment_id}"
            )
            raise click.Abort()

    # If no existing experiment, create a new one
    click.echo(f"Creating new experiment: '{experiment_name}'")
    experiment_id = mlflow.create_experiment(name=experiment_name)
    click.echo(f"Experiment '{experiment_name}' created with ID {experiment_id}.")

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
        click.echo("ℹ️  Please create an experiment using `kedro experiment create` before proceeding.")
        raise click.Abort()
