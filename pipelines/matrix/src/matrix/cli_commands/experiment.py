import os
from typing import Optional

import click
import mlflow
from kedro.framework.cli.project import (
    PIPELINE_ARG_HELP,
)
from kedro.framework.cli.utils import (
    env_option,
)

from matrix.cli_commands.submit import submit
from matrix.git_utils import create_new_branch, get_current_git_branch
from matrix.utils.authentication import get_iap_token


@click.group()
def experiment():
    pass


@experiment.command()
def create():
    current_branch = get_current_git_branch()

    # TODO: Change to experiment/
    if current_branch.startswith("feat/"):
        click.confirm(f"Current experiment is running from branch {current_branch}, is that correct?")
        exp_name = current_branch.strip("feat/")
    elif current_branch == "main":
        exp_name = click.prompt("Please enter a name for your experiment", type=str)
        create_new_branch(exp_name)
    else:
        print("Error: please begin with an experiment/ branch or begin from main")
        click.Abort()

    branch_name = f"experiment/{exp_name}"
    print("branch_name", branch_name)
    print("exp_name", exp_name)

    mlflow_id = create_mlflow_experiment(experiment_name=exp_name)

    click.echo(f"kedro experiment created on branch {branch_name}")
    click.echo(f"mlflow experiment created: {mlflow.get_tracking_uri()}/#/experiments/{mlflow_id}")

    print("experiment group created")


@experiment.command()
@env_option
# TODO: add all requried options
# @click.argument( "function_to_call",      type=str, default=None,)
@click.option(
    "--experiment_name",
    type=str,
    default=None,
)
@click.option("--pipeline", "-p", required=True, default="__default__", type=str, help=PIPELINE_ARG_HELP)
@click.option("--username", type=str, required=True, help="Specify the username to use")
@click.option("--release-version", type=str, required=True, help="Specify a custom release name")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
# fmt: on
# TODO: remove this - we won't need to pass context anymore. just point directly at submit
# Unless we need to maintain submit...
@click.pass_context
def run(
    ctx, env: str, experiment_name: Optional[str], pipeline: str, username: str, release_version: str, dry_run: bool
):
    """Run an experiment."""
    if not experiment_name:
        # TODO: sanitize branch name
        experiment_name = get_current_git_branch()
        print("experiment_name", experiment_name)

    run_id = get_run_id_from_mlflow_name(experiment_name=experiment_name)

    ctx.invoke(
        submit,
        username=username,
        release_version=release_version,
        pipeline=pipeline,
        experiment_id=run_id,
        dry_run=dry_run,
    )


def create_mlflow_experiment(experiment_name: str):
    token = get_iap_token()
    mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token

    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if experiment:
        click.echo(
            f"Error: Experiment {experiment_name} already found at {mlflow.get_tracking_uri()}/#/experiments/{experiment.experiment_id}"
        )
        raise click.Abort()

    else:
        click.echo(f"Generating experiment {experiment_name}")
        # TODO: Add error handling
        experiment_id = mlflow.create_experiment(name=experiment_name)
        click.echo(f"Experiment {experiment_name} generated with ID {experiment_id}")

        return experiment_id


def get_run_id_from_mlflow_name(experiment_name: str):
    token = get_iap_token()
    # TODO: Pull from config?
    mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token

    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if experiment:
        click.echo(
            f"Experiment {experiment_name} found at {mlflow.get_tracking_uri()}/#/experiments/{experiment.experiment_id}"
        )
        return experiment.experiment_id

    else:
        click.echo(f"Experiment {experiment_name} not found. Please create an experiment first.")
        raise click.Abort()
