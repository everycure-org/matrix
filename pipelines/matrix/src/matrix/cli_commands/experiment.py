import os
import secrets
from typing import List

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
def create():
    current_branch = get_current_git_branch()
    # temp: current branch is feat/, but want to test with experiment/
    current_branch = current_branch.replace("feat/", "experiment/")

    # TODO: Change to experiment/
    if current_branch.startswith("experiment/"):
        click.confirm(
            f"Creating new mlflow experiment from current branch {current_branch}, is that correct?", abort=True
        )
        exp_name = current_branch
    elif current_branch == "main":
        user_input_name = click.prompt("Please enter a name for your experiment", type=str)
        if not user_input_name.startswith("experiment/"):
            exp_name = f"experiment/{user_input_name}"
        create_new_branch(exp_name)
    else:
        print("Error: please begin with an experiment/ branch or begin from main")
        raise click.Abort()

    mlflow_id = create_mlflow_experiment(experiment_name=exp_name)

    click.echo(f"kedro experiment created on branch {exp_name}")
    click.echo(f"mlflow experiment created: {mlflow.get_tracking_uri()}/#/experiments/{mlflow_id}")

    print("experiment group created")


@experiment.command()
# @env_option
# These are all copied directly from submit. If we want to maintain kedro submit functionality I think we need to
# keep the duplication for now. Then we can just rename submit to run and add the extra mlflow steps there.
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

    current_branch = get_current_git_branch()
    # temp: current branch is feat/, but want to test with experiment/
    experiment_name = current_branch.replace("feat/", "experiment/")

    click.confirm(f"Start a new run on experiment {experiment_name}, is that correct?", abort=True)
    experiment_id = get_run_id_from_mlflow_name(experiment_name=experiment_name)

    if not run_name:
        # Note, it is not possible to search mlflow runs by name, so we cannot enforce uniqueness, this is down to the user
        run_name = click.prompt("Please define a name for your run")

    # Invokes the the submit command and forwards all arguments from current command
    # Temporary measure until we formally deprecate kedro submit
    ctx.forward(submit, experiment_id=experiment_id, run_name=run_name)


def create_mlflow_experiment(experiment_name: str) -> str:
    existing_experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if existing_experiment:
        if existing_experiment.lifecycle_stage == "deleted":
            click.echo(f"Error: Experiment {experiment_name} already exists and has been deleted.")

            if click.confirm(
                f"Would you like to rename the deleted experiment and continue with this experiment name: {experiment_name}?"
            ):
                try:
                    mlflow.tracking.MlflowClient().restore_experiment(existing_experiment.experiment_id)
                    random_sfx = str.lower(secrets.token_hex(4))
                    mlflow.tracking.MlflowClient().rename_experiment(
                        experiment_id=existing_experiment.experiment_id,
                        new_name=f"deleted-{experiment_name}-{random_sfx}",
                    )
                    mlflow.delete_experiment(existing_experiment.experiment_id)

                    experiment_id = mlflow.create_experiment(name=experiment_name)
                except Exception as e:
                    click.echo(e)
                    raise click.Abort()
            else:
                new_exp_name = click.prompt("Please provide a new experiment name")
                print(new_exp_name)
                experiment_id = mlflow.create_experiment(name=new_exp_name)

            return experiment_id

        else:
            click.echo(
                f"Error: Experiment {experiment_name} already found at {mlflow.get_tracking_uri()}/#/experiments/{existing_experiment.experiment_id}"
            )
            raise click.Abort()

    else:
        click.echo(f"Generating experiment {experiment_name}")
        experiment_id = mlflow.create_experiment(name=experiment_name)
        click.echo(f"Experiment {experiment_name} generated with ID {experiment_id}")

        return experiment_id


def get_run_id_from_mlflow_name(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if experiment:
        click.echo(
            f"Experiment {experiment_name} found at {mlflow.get_tracking_uri()}/#/experiments/{experiment.experiment_id}"
        )
        return experiment.experiment_id

    else:
        click.echo(f"Experiment {experiment_name} not found. Please create an experiment first.")
        raise click.Abort()
