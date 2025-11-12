import json
import logging
import os
import re
import secrets
import sys
from pathlib import Path
from typing import List, Optional

import click
import mlflow
from kedro.framework.cli.utils import split_string
from kedro.framework.project import pipelines as kedro_pipelines
from kedro.framework.startup import bootstrap_project
from kedro.pipeline import Pipeline
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from matrix.argo import ARGO_TEMPLATES_DIR_PATH, generate_argo_config
from matrix.cli_commands.run import _validate_env_vars_for_private_data
from matrix.utils.environment import load_environment_variables

# Load environment variables from .env.defaults and .env
load_environment_variables()

from matrix_auth.authentication import get_user_account_creds
from matrix_mlflow_utils.mlflow_utils import (
    DeletedExperimentExistsWithName,
    ExperimentNotFound,
    archive_runs_and_experiments,
    create_mlflow_experiment,
    get_experiment_id_from_name,
    rename_soft_deleted_experiment,
)

from matrix.git_utils import (
    BRANCH_NAME_REGEX,
    abort_if_intermediate_release,
    get_changed_git_files,
    get_current_git_branch,
    get_current_git_sha,
    git_tag_exists,
    has_legal_branch_name,
    has_unpushed_commits,
)
from matrix.utils.argo import argo_template_lint, submit_workflow
from matrix.utils.kubernetes import apply, can_talk_to_kubernetes, create_namespace, namespace_exists
from matrix.utils.system import run_subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")
console = Console()

EXPERIMENT_BRANCH_PREFIX = "experiment/"


def configure_mlflow_tracking(token: str) -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_URL"])
    os.environ["MLFLOW_TRACKING_TOKEN"] = token


def get_user_account_token() -> str:
    try:
        return get_user_account_creds().id_token
    except FileNotFoundError as e:
        click.secho("Error getting IAP token. Please run `make fetch_secrets` first", fg="yellow", bold=True)
        raise


@click.group()
def experiment() -> None:
    _validate_env_vars_for_private_data()
    if os.getenv("GITHUB_ACTIONS") and os.getenv("GCP_TOKEN"):
        # Running in GitHub Actions, get the IAP token of service acccount from the secrets
        click.echo("Running in GitHub Actions, using service account IAP token")
        token = os.getenv("GCP_TOKEN")
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
    "--confirm-release", is_flag=True, type=bool, default=False, help="Confirm that we want to submit a release"
)
@click.option(
    "--experiment-name", type=str, help="Optional: specify the MLFlow experiment name to use. Defaults to branch name"
)
@click.option(
    "--from-run",
    type=str,
    default=None,
    help="Run to read from, if specified will read from the `--from-run` datasets and write to the run_name datasets",
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
    confirm_release: bool,
    experiment_name: str,
    from_run: Optional[str] = None,
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

    run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
    mlflow.set_tag("created_by", username)
    mlflow_run_id = run.info.run_id

    if not quiet:
        log.setLevel(logging.DEBUG)

    if pipeline in ("data_release", "kg_release_and_matrix_run", "kg_release_patch_and_matrix_run"):
        if not headless and not confirm_release:
            if not click.confirm(
                "Manual release submission detected, releases must be submitted via the release pipeline. Are you sure you want to create a manual release?",
                default=False,
            ):
                raise click.Abort()

        if not skip_git_checks:
            abort_if_unmet_git_requirements(release_version)
            abort_if_intermediate_release(release_version)

    if pipeline not in kedro_pipelines.keys():
        raise ValueError("Pipeline requested for execution not found")

    if pipeline in ["fabricator", "test"]:
        raise ValueError("Submitting test pipeline to Argo will result in overwriting source data")

    if not headless and (from_nodes or nodes):
        if not click.confirm(
            "Using 'from-nodes' or 'nodes' is highly experimental and may break due to MLFlow issues with tracking the right run. Are you sure you want to continue?",
            default=False,
        ):
            raise click.Abort()

    pipeline_obj = kedro_pipelines[pipeline]
    if from_nodes:
        pipeline_obj = pipeline_obj.from_nodes(*from_nodes)

    if nodes:
        pipeline_obj = pipeline_obj.filter(node_names=nodes)

    run_name = get_run_name(run_name)
    pipeline_obj.name = pipeline

    if not dry_run:
        summarize_submission(
            experiment_id, run_name, namespace, pipeline, environment, is_test, release_version, headless
        )

    argo_url = _submit(
        username=username,
        namespace=namespace,
        run_name=run_name,
        release_version=release_version,
        pipeline_obj=pipeline_obj,
        verbose=not quiet,
        dry_run=dry_run,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
        mlflow_experiment_id=experiment_id,
        mlflow_run_id=mlflow_run_id,
        allow_interactions=not headless,
        is_test=is_test,
        environment=environment,
        from_run=from_run,
    )

    # construct description for mlflow run which we'll use to add some useful links
    description = f"""
- [Argo Workflow]({argo_url})
- [Codebase](https://github.com/everycure-org/matrix/tree/{get_current_git_sha()})
    """

    # see https://stackoverflow.com/questions/73320708/set-run-description-programmatically-in-mlflow
    # or https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html
    mlflow.set_tag("mlflow.note.content", description)


@experiment.command()
@click.option("--run-name", type=str, required=True, help="Specify the run name to use")
@click.option("--watch", is_flag=True, default=False, help="Watch the workflow after retrying")
@click.option("--workflow-file", type=str, help="Specify the workflow file to use, instead of calling kubectl")
def retry(run_name: str, watch: bool, workflow_file: Optional[str]):
    # Grab workflow document from kubectl or from file
    console.print("1. Grabbing the image name from the workflow spec")
    if workflow_file:
        workflow_doc = Path(workflow_file).read_text()
    else:
        workflow_doc = run_subprocess(
            f"kubectl get workflow -n argo-workflows -o json {run_name}",
        ).stdout.strip()
    workflow_dict = json.loads(workflow_doc)

    # grab all existing arguments and overwrite the image one
    parameters = workflow_dict["status"]["storedWorkflowTemplateSpec"]["arguments"]["parameters"]
    for par in parameters:
        if par["name"] == "image":
            image_name = par["value"]
            assert "matrix-images" in image_name, f"Image name seems to be incorrect {image_name}"
            image_without_tag = image_name.split(":")[0]
            new_image_tag = f"{'-'.join(run_name.split('-')[:-1])}-rerun{os.urandom(4).hex()}"
            new_image_name = f"{image_without_tag}:{new_image_tag}"
            par["value"] = new_image_name
            break

    # convert params into '-p key=val pairs'
    parameters_str = " ".join([f"-p {arg['name']}={arg['value']}" for arg in parameters])

    console.print("2. Re-building the docker image")
    build_push_docker(image_without_tag, new_image_tag, verbose=False)
    # call argo retry
    console.print("3. Calling Argo Retry")
    watch_flag = "--watch" if watch else ""
    run_subprocess(f"argo retry -n argo-workflows {parameters_str} {watch_flag} {run_name}")
    console.print(f"4. Workflow {run_name} retried successfully")


@experiment.command()
@click.option("--dry-run", "-d", is_flag=True, help="Whether to actually archive runs and experiments")
def archive(dry_run: bool):
    if dry_run:
        click.echo("Dry run of archiving runs and experiments")
        archive_runs_and_experiments(dry_run=dry_run)
    elif click.confirm("Are you sure you want to archive all runs and experiments?", abort=True):
        archive_runs_and_experiments(dry_run=dry_run)


def _submit(
    username: str,
    namespace: str,
    run_name: str,
    release_version: str,
    pipeline_obj: Pipeline,
    verbose: bool,
    dry_run: bool,
    environment: str,
    template_directory: Path,
    mlflow_experiment_id: int,
    mlflow_run_id: Optional[str] = None,
    allow_interactions: bool = True,
    is_test: bool = False,
    from_run: Optional[str] = None,
) -> str:
    """Submit the end-to-end workflow.

    This function contains redundancy.

    The original logic of this function was:
    1. Create & Apply (push to k8s) Argo template, containing the entire pipeline registry. This part of the function makes use of pipelines_for_workflow, which will be included in the template.
    2. When submitting the workflow, via `__entrypoint__`, a pipeline for execution is selected.
        It defaults to `__default__`, but can be configured via pipeline_for_execution.

    In the future, we expect plan not to have any template at all, but go straight from Kedro to Argo Workflow.

    This meant that it was possible to submit the workflows for other pipelines in Argo CI.

    Args:
        username (str): The username to use for the workflow.
        namespace (str): The namespace to use for the workflow.
        run_name (str): The name of the run.
        pipeline_obj (Pipeline): Pipeline to execute.
        verbose (bool): If True, enable verbose output.
        dry_run (bool): If True, do not submit the workflow.
        template_directory (Path): The directory containing the Argo template.
        allow_interactions (bool): If True, allow prompts for confirmation.
        is_test (bool): If True, submit to test folder, not release folder.
    """

    try:
        runtime_gcp_project_id = os.environ["RUNTIME_GCP_PROJECT_ID"]
        mlflow_url = os.environ["MLFLOW_URL"]
        image = f"us-central1-docker.pkg.dev/{runtime_gcp_project_id}/matrix-images/matrix"

        console.rule("[bold blue]Submitting Workflow")
        if not can_talk_to_kubernetes(
            project=runtime_gcp_project_id, region="us-central1", cluster_name="compute-cluster"
        ):
            raise EnvironmentError("Cannot communicate with Kubernetes")
        else:
            console.print("[green]✓[/green] kubectl authenticated")

        argo_template = build_argo_template(
            f"{image}:{run_name}",
            run_name,
            release_version,
            username,
            namespace,
            pipeline_obj,
            environment,
            mlflow_experiment_id,
            mlflow_url,
            is_test=is_test,
            mlflow_run_id=mlflow_run_id,
            from_run=from_run,
        )

        file_path = save_argo_template(argo_template, template_directory)
        console.print("Linting Argo template...")
        argo_template_lint(file_path, verbose=verbose)
        console.print("[green]✓[/green] Argo template linted")

        if dry_run:
            return

        console.print("Building Docker image...")
        build_push_docker(image, run_name, verbose=verbose)
        console.print("[green]✓[/green] Docker image built and pushed to dev repository")

        console.print("Ensuring Kubernetes namespace...")
        if not namespace_exists(namespace):
            console.print(f"Namespace {namespace} does not exist. Creating it...")
            create_namespace(namespace, verbose=verbose)
        else:
            console.print("[green]✓[/green] Namespace exists")

        apply(namespace, file_path, verbose=verbose)
        console.print("[green]✓[/green] Argo template applied")

        console.print("Submitting workflow for pipeline...")
        job_name = submit_workflow(run_name, namespace, verbose=verbose)
        argo_url = f"{os.environ['ARGO_PLATFORM_URL']}/workflows/argo-workflows/{run_name}"
        console.print(f"\nSee your workflow in the ArgoCD UI here: [blue]{argo_url}[/blue]")
        console.print(f"Workflow submitted successfully with job name: {job_name}")
        console.print("\nTo watch the workflow progress, run the following command:")
        console.print(f"argo watch -n {namespace} {job_name}")
        console.print("\nTo view the workflow in the Argo UI, run:")
        console.print(f"argo get -n {namespace} {job_name}")
        console.print("[green]✓[/green] Workflow submitted")

        console.print(
            Panel.fit(
                f"[bold green]Workflow {'prepared' if dry_run else 'submitted'} successfully![/bold green]\n"
                f"Run Name: {run_name}\n"
                f"Namespace: {namespace}",
                title="Submission Summary",
            )
        )

        if allow_interactions and click.confirm("Do you want to open the workflow in your browser?", default=False):
            workflow_url = f"{os.environ['ARGO_PLATFORM_URL']}/workflows/{namespace}/{run_name}"
            click.launch(workflow_url)
            console.print(f"[blue]Opened workflow in browser: {workflow_url}[/blue]")

        return argo_url

    except Exception as e:
        console.print(f"[bold red]Error during submission:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def summarize_submission(
    experiment_id: int,
    run_name: str,
    namespace: str,
    pipeline: str,
    environment: str,
    is_test: bool,
    release_version: str,
    headless: bool,
):
    console.print(
        Panel.fit(
            f"[bold green]About to submit workflow:[/bold green]\n"
            f"MLFlow Experiment: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}\n"
            f"Run Name: {run_name}\n"
            f"Namespace: {namespace}\n"
            f"Pipeline: {pipeline}\n"
            f"Environment: {environment}\n"
            f"Writing to test folder: {is_test}\n"
            f"Data Release Version: {release_version}\n",
            title="Submission Summary",
        )
    )
    console.print(
        "Reminder: A data release should only be submitted once and not overwritten.\n"
        "If you need to make changes, please make this part of the next release.\n"
        "Experiments (modelling pipeline) are nested under the release and can be overwritten.\n\n"
    )

    if not headless:
        if not click.confirm("Are you sure you want to submit the workflow?", default=False):
            raise click.Abort()


def build_push_docker(image: str, username: str, verbose: bool):
    """Build the docker image only once, push it to dev registry, and if running in prod, also to prod registry."""
    run_subprocess(f"make docker_push TAG={username} docker_image={image}", stream_output=verbose)


def build_argo_template(
    image_name: str,
    run_name: str,
    release_version: str,
    username: str,
    namespace: str,
    pipeline_obj: Pipeline,
    environment: str,
    mlflow_experiment_id: int,
    mlflow_url: str,
    is_test: bool,
    mlflow_run_id: Optional[str] = None,
    from_run: Optional[str] = None,
) -> str:
    """Build Argo workflow template."""
    matrix_root = Path(__file__).parent.parent.parent.parent
    metadata = bootstrap_project(matrix_root)
    package_name = metadata.package_name

    if is_test:
        release_folder_name = "tests"
    else:
        release_folder_name = "releases"

    console.print("Building Argo template...")
    generated_template = generate_argo_config(
        image=image_name,
        run_name=run_name,
        release_version=release_version,
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_url=mlflow_url,
        namespace=namespace,
        username=username,
        package_name=package_name,
        release_folder_name=release_folder_name,
        pipeline=pipeline_obj,
        environment=environment,
        mlflow_run_id=mlflow_run_id,
        from_run=from_run,
    )
    console.print("[green]✓[/green] Argo template built")

    return generated_template


def save_argo_template(argo_template: str, template_directory: Path) -> str:
    console.print("Writing Argo template...")
    file_path = template_directory / "argo-workflow-template.yml"
    with open(file_path, "w") as f:
        f.write(argo_template)
    console.print(f"[green]✓[/green] Argo template saved to {file_path}")
    return str(file_path)


def get_run_name(run_name: Optional[str]) -> str:
    """Get the experiment name based on input or Git branch.

    If a run_name is provided, it is returned sanitized as-is. Otherwise, a name is generated
    based on the current Git branch name with a random suffix.

    Args:
        run_name (Optional[str]): A custom run name provided by the user.

    Returns:
        str: The final run name to be used for the workflow.
    """
    # If no run_name is provided, use the current Git branch name
    if not run_name:
        run_name = run_subprocess("git rev-parse --abbrev-ref HEAD", stream_output=True).stdout.strip()

    # Add a random suffix to the run_name
    random_sfx = str.lower(secrets.token_hex(4))
    unsanitized_name = f"{run_name}-{random_sfx}".rstrip("-")
    sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", unsanitized_name)
    return sanitized_name


def abort_if_unmet_git_requirements(release_version: str) -> None:
    """
    Validates the current Git repository:
    1. The current Git branch must be either 'main' or 'master'.
    2. The Git repository must be clean (no uncommitted changes or untracked files).

    Raises:
        ValueError
    """
    errors = []

    if len(get_changed_git_files()) > 0:
        errors.append(f"Repository has uncommitted changes or untracked files: {';'.join(get_changed_git_files())}")

    if not has_legal_branch_name():
        errors.append(f"Your branch name doesn't match the regex: {BRANCH_NAME_REGEX}")

    if has_unpushed_commits():
        errors.append(f"You have commits not pushed to remote.")

    if git_tag_exists(release_version):
        errors.append(f"The git tag for the release version you specified already exists.")

    if errors:
        error_list = "\n".join(errors)
        raise RuntimeError(f"Submission failed due to the following issues:\n\n{error_list}")
