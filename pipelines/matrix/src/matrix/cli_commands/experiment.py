import os
from typing import List, Optional

import click
import mlflow
from kedro.framework.cli.utils import split_string

from matrix.cli_commands.submit import submit
from matrix.git_utils import get_current_git_branch
from matrix.utils.authentication import get_iap_token
from matrix.utils.mlflow_utils import (
    DeletedExperimentExistsWithName,
    ExperimentNotFound,
    create_mlflow_experiment,
    get_experiment_id_from_name,
    rename_soft_deleted_experiment,
)

EXPERIMENT_BRANCH_PREFIX = "experiment/"


@click.group()
def experiment():
    try:
        token = get_iap_token()
        mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
        os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token
    except FileNotFoundError as e:
        click.secho("Error getting IAP token. Please run `make fetch_secrets` first", fg="yellow", bold=True)
        raise click.Abort()
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
                f"✅ Deleted experiment {experiment_name} has been renamed to {renamed_exp}. You may now `kedro experiment create` using this name."
            )
            raise click.Abort()

    click.echo(f"✅ MLFlow experiment created: {mlflow.get_tracking_uri()}/#/experiments/{mlflow_id}")


@experiment.command()
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
        experiment_name = get_current_git_branch()
        if not experiment_name.startswith(EXPERIMENT_BRANCH_PREFIX):
            click.echo(
                f"❌ Error: current branch does not begin with experiment/. Please define an experiment name or start from an experiment branch."
            )
            raise click.Abort()
        experiment_name = experiment_name.strip(EXPERIMENT_BRANCH_PREFIX)

    click.confirm(f"Start a new run on experiment '{experiment_name}', is that correct?", abort=True)

    try:
        experiment_id = get_experiment_id_from_name(experiment_name=experiment_name)
    except ExperimentNotFound:
        if click.confirm(
            f"Experiment '{experiment_name}' not found. Would you like to create a new experiment?", abort=True
        ):
            experiment_id = create_mlflow_experiment(experiment_name=experiment_name)

    if not run_name:
        run_name = click.prompt("Please define a name for your run")

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
        skip_git_checks=skip_git_checks,
    )
