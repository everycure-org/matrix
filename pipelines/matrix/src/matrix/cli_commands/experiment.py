import os
from typing import Optional

import click
import mlflow
from kedro.framework.cli.project import (
    PIPELINE_ARG_HELP,
    project_group,
)
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    KedroCliError,
    env_option,
)

from matrix.cli_commands.submit import submit
from matrix.git_utils import get_current_git_branch
from matrix.utils.authentication import get_iap_token


@click.group()
def experiment():
    print("here")
    pass


@experiment.command()
def create():
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
@click.pass_context
def run(
    ctx, env: str, experiment_name: Optional[str], pipeline: str, username: str, release_version: str, dry_run: bool
):
    """Run an experiment."""
    if not experiment_name:
        # TODO: sanitize branch name
        experiment_name = get_current_git_branch()
        print("experiment_name", experiment_name)

    run_id = get_run_id_from_mlflow(experiment_name=experiment_name)

    ctx.invoke(
        submit,
        username=username,
        release_version=release_version,
        pipeline=pipeline,
        experiment_id=run_id,
        dry_run=dry_run,
    )


def get_run_id_from_mlflow(experiment_name: str):
    token = get_iap_token()
    # TODO: Pull from config?
    mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token

    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Experiment {experiment_name} found with ID {experiment_id}")
    else:
        print(f"Experiment {experiment_name} not found. Generating...")
        # TODO: Add error handling
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Experiment {experiment_name} generated with ID {experiment_id}")

    return experiment_id
