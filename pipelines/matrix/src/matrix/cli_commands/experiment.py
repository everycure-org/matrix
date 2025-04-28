import json
import os
import re
from typing import List, Literal

import click
import mlflow
from kedro.framework.cli.utils import split_string

from matrix.cli_commands.run import _validate_env_vars_for_private_data
from matrix.cli_commands.submit import submit
from matrix.git_utils import get_current_git_branch
from matrix.utils.authentication import get_service_account_creds, get_user_account_creds
from matrix.utils.mlflow_utils import (
    DeletedExperimentExistsWithName,
    ExperimentNotFound,
    archive_runs_and_experiments,
    create_mlflow_experiment,
    get_experiment_id_from_name,
    rename_soft_deleted_experiment,
)

EXPERIMENT_BRANCH_PREFIX = "experiment/"


def configure_mlflow_tracking(token: str) -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_URL"])
    os.environ["MLFLOW_TRACKING_TOKEN"] = token


def get_service_account_token() -> str:
    try:
        sa_credential_info = json.loads(os.getenv("GCP_SA_KEY"))
        return get_service_account_creds(sa_credential_info).token
    except json.JSONDecodeError as e:
        click.secho(
            "Error decoding service account key. Please check the format and presence of the GCP_SA_KEY secret",
            fg="yellow",
            bold=True,
        )
        raise


def get_user_account_token() -> str:
    try:
        return get_user_account_creds().id_token
    except FileNotFoundError as e:
        click.secho("Error getting IAP token. Please run `make fetch_secrets` first", fg="yellow", bold=True)
        raise


@click.group()
def experiment() -> None:
    _validate_env_vars_for_private_data()
    if os.getenv("GITHUB_ACTIONS"):
        # Running in GitHub Actions, get the IAP token of service acccount from the secrets
        click.echo("Running in GitHub Actions, using service account IAP token")
        token = get_service_account_token()
    else:
        # Running locally, get the IAP token of user account
        token = get_user_account_token()
    configure_mlflow_tracking(token)


@experiment.command()
@click.option(
    "--experiment-name",
    type=str,
    help="Optional: specify the experiment name to use. Otherwise, default to branch name",
)
def create(experiment_name):
    if not experiment_name:
        current_branch = get_current_git_branch()
        if current_branch.startswith(EXPERIMENT_BRANCH_PREFIX):
            click.confirm(
                f"Creating new mlflow experiment from current branch {current_branch}, is that correct?", abort=True
            )
            experiment_name = current_branch.strip(EXPERIMENT_BRANCH_PREFIX)
        else:
            experiment_name = click.prompt("Please enter a name for your experiment", type=str)

    try:
        mlflow_id = create_mlflow_experiment(experiment_name=experiment_name)
    except DeletedExperimentExistsWithName as e:
        if click.confirm(
            f"Would you like to rename the deleted experiment and continue using the name '{experiment_name}'?",
            abort=True,
        ):
            renamed_exp = rename_soft_deleted_experiment(experiment_name)
            click.echo(
                f"✅ Deleted experiment {experiment_name} has been renamed to {renamed_exp}. You may now retry using this name."
            )
            raise click.Abort()

    click.echo(f"✅ MLFlow experiment created: {mlflow.get_tracking_uri()}/#/experiments/{mlflow_id}")
    return mlflow_id


# These are all copied directly from submit. If we want to maintain kedro submit functionality I think we need to
# keep the duplication for now. Then we can just rename submit to run and add the extra mlflow steps.
@experiment.command()
@click.option("--username", type=str, required=True, help="Specify the username to use")
@click.option("--namespace", type=str, default="argo-workflows", help="Specify a custom namespace")
@click.option("--run-name", type=str, default=None, help="Specify a custom run name, defaults to branch")
@click.option("--release-version", type=str, required=True, help="Specify a custom release name")
@click.option("--pipeline", "-p", type=str, default="modelling_run", help="Specify which pipeline to execute")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Disable verbose output")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
@click.option("--from-nodes", type=str, default="", help="Specify nodes to run from", callback=split_string)
@click.option("--nodes", "-n", type=str, default="", help="Specify nodes to run", callback=split_string)
@click.option("--is-test", is_flag=True, default=False, help="Submit to test folder")
@click.option("--headless", is_flag=True, default=False, help="Skip confirmation prompt")
@click.option("--environment", "-e", type=str, default="cloud", help="Kedro environment to execute in")
@click.option("--skip-git-checks", is_flag=True, type=bool, default=False, help="Skip git checks")
@click.option(
    "--experiment-name", type=str, help="Optional: specify the MLFlow experiment name to use. Defaults to branch name"
)
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
    nodes: List[str],
    is_test: bool,
    headless: bool,
    environment: str,
    skip_git_checks: bool,
    experiment_name: str,
):
    """Run an experiment."""

    if not experiment_name:
        current_branch = get_current_git_branch()
        sanitized_branch_name = re.sub(r"[^a-zA-Z0-9-]", "-", current_branch)
        if headless or click.confirm(
            f"Would you like to use the current branch '{sanitized_branch_name}' as the experiment name?"
        ):
            experiment_name = sanitized_branch_name
        else:
            experiment_name = click.prompt("Please enter a name for your experiment", type=str)

    try:
        experiment_id = get_experiment_id_from_name(experiment_name=experiment_name)
    except ExperimentNotFound:
        if not headless:
            click.confirm(
                f"Experiment '{experiment_name}' not found, would you like to create a new experiment?", abort=True
            )
        experiment_id = ctx.invoke(create, experiment_name=experiment_name)

    if not run_name:
        run_name = click.prompt("Please define a name for your run")

    if not headless:
        click.confirm(f"Start a new run '{run_name}' on experiment '{experiment_name}', is that correct?", abort=True)

    if not dry_run:
        run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
        mlflow.set_tag("created_by", username)
        mlflow_run_id = run.info.run_id

    ctx.invoke(
        submit,
        username=username,
        namespace=namespace,
        run_name=run_name,
        release_version=release_version,
        pipeline=pipeline,
        quiet=quiet,
        dry_run=dry_run,
        from_nodes=from_nodes,
        nodes=nodes,
        is_test=is_test,
        headless=headless,
        environment=environment,
        experiment_id=experiment_id,
        mlflow_run_id=mlflow_run_id,
        skip_git_checks=skip_git_checks,
    )


@experiment.command()
@click.option("--dry-run", "-d", is_flag=True, help="Whether to actually archive runs and experiments")
def archive(dry_run: bool):
    if dry_run:
        click.echo("Dry run of archiving runs and experiments")
        archive_runs_and_experiments(dry_run=dry_run)
    elif click.confirm("Are you sure you want to archive all runs and experiments?", abort=True):
        archive_runs_and_experiments(dry_run=dry_run)
