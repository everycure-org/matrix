import json
import logging
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS, split_string
from kedro.framework.project import pipelines as kedro_pipelines
from kedro.framework.startup import bootstrap_project
from kedro.pipeline import Pipeline
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from matrix.argo import ARGO_TEMPLATES_DIR_PATH, generate_argo_config
from matrix.kedro4argo_node import ArgoResourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


console = Console()


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


# fmt: off
@cli.command()
@click.option("--username", type=str, required=True, help="Specify the username to use")
@click.option("--namespace", type=str, default="argo-workflows", help="Specify a custom namespace")
@click.option("--run-name", type=str, default=None, help="Specify a custom run name, defaults to branch")
@click.option("--release-version", type=str, required=True, help="Specify a custom release name")
@click.option("--pipeline", type=str, default="__default__", help="Specify which pipeline to execute")
@click.option("--verbose", "-v", is_flag=True, default=True, help="Enable verbose output")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
@click.option("--from-nodes", type=str, default="", help="Specify nodes to run from", callback=split_string)
@click.option("--is-test", is_flag=True, default=False, help="Submit to test folder")
# fmt: on
def submit(username: str, namespace: str, run_name: str, release_version: str, pipeline: str, verbose: bool, dry_run: bool, from_nodes: List[str], is_test: bool):
    """Submit the end-to-end workflow. """
    if verbose:
        log.setLevel(logging.DEBUG)

    if pipeline not in kedro_pipelines.keys():
        raise ValueError("Pipeline requested for execution not found")
    
    if pipeline in ["fabricator", "test"]:
        raise ValueError("Submitting test pipeline to Argo will result in overwriting source data")
    
    if from_nodes:
        if not click.confirm("Using 'from-nodes' is highly experimental and may break due to MLFlow issues with tracking the right run. Are you sure you want to continue?", default=False):
            raise click.Abort()
    
    # As a temporary measure, we pass both pipeline for execution and list of pipelines. In the future, we will merge the two.
    pipeline_obj = kedro_pipelines[pipeline]
    if from_nodes:
        pipeline_obj = pipeline_obj.from_nodes(*from_nodes)

    if not run_name:
        run_name = get_run_name(run_name)

    pipeline_obj.name = pipeline


    _submit(
        username=username,
        namespace=namespace,
        run_name=run_name,
        release_version=release_version,
        pipeline_obj=pipeline_obj,
        verbose=verbose,
        dry_run=dry_run,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
        is_test=is_test,
    )


def _submit(
        username: str, 
        namespace: str, 
        run_name: str, 
        release_version: str,
        pipeline_obj: Pipeline,
        verbose: bool, 
        dry_run: bool, 
        template_directory: Path,
        allow_interactions: bool = True,
        is_test: bool = False,
    ) -> None:
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
        allow_interactions (bool): If True, allow prompts for confirmation
        is_test (bool): If True, submit to test folder, not release folder
    """
    
    try:
        console.rule("[bold blue]Submitting Workflow")

        console.print("Checking dependencies...")
        check_dependencies(verbose=verbose)
        console.print("[green]✓[/green] Dependencies checked")

        console.print("Building Argo template...")
        argo_template = build_argo_template(run_name, release_version, username, namespace, pipeline_obj, is_test=is_test, )
        console.print("[green]✓[/green] Argo template built")

        console.print("Writing Argo template...")
        file_path = save_argo_template(argo_template, template_directory)
        console.print("[green]✓[/green] Argo template written")

        console.print("Linting Argo template...")
        argo_template_lint(file_path, verbose=verbose)
        console.print("[green]✓[/green] Argo template valid")

        if not dry_run:
            console.print("Building and pushing Docker image...")
            build_push_docker(run_name, verbose=verbose)
            console.print("[green]✓[/green] Docker image built and pushed")

            console.print("Ensuring namespace...")
            ensure_namespace(namespace, verbose=verbose)
            console.print("[green]✓[/green] Namespace ensured")

            console.print("Applying Argo template...")
            apply_argo_template(namespace, file_path, verbose=verbose)
            console.print("[green]✓[/green] Argo template applied")

            console.print("Submitting workflow for pipeline...")
            submit_workflow(run_name, namespace, verbose=verbose)
            console.print("[green]✓[/green] Workflow submitted")

            console.print(Panel.fit(
                f"[bold green]Workflow {'prepared' if dry_run else 'submitted'} successfully![/bold green]\n"
                f"Run Name: {run_name}\n"
                f"Namespace: {namespace}",
                title="Submission Summary"
            ))

        if not dry_run and allow_interactions and click.confirm("Do you want to open the workflow in your browser?", default=False):
            workflow_url = f"https://argo.platform.dev.everycure.org/workflows/{namespace}/{run_name}"
            click.launch(workflow_url)
            console.print(f"[blue]Opened workflow in browser: {workflow_url}[/blue]")
    except Exception as e:
        console.print(f"[bold red]Error during submission:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


        

def run_subprocess(
    cmd: str,
    check: bool = True,
    capture_output: bool = True,
    shell: bool = True,
    stream_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and handle errors.

    :param cmd: Command string to execute
    :param check: If True, raise CalledProcessError on non-zero exit status
    :param capture_output: If True, capture stdout and stderr
    :param shell: If True, execute the command through the shell
    :param stream_output: If True, stream output to stdout and stderr
    :return: CompletedProcess instance
    """
    if stream_output:
        process = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        stdout, stderr = [], []
        for line in process.stdout:
            sys.stdout.write(line)
            stdout.append(line)
        for line in process.stderr:
            sys.stderr.write(line)
            stderr.append(line)

        returncode = process.wait()
        if check and returncode != 0:
            raise subprocess.CalledProcessError(
                returncode, cmd, "".join(stdout), "".join(stderr)
            )

        return subprocess.CompletedProcess(
            cmd, returncode, "".join(stdout), "".join(stderr)
        )
    else:
        try:
            return subprocess.run(
                cmd, check=check, capture_output=capture_output, text=True, shell=shell
            )
        except subprocess.CalledProcessError as e:
            console.print(f"Error executing command: {cmd}")
            console.print(f"Exit code: {e.returncode}")
            if e.stdout:
                console.print(f"stdout: {e.stdout}")
            if e.stderr:
                console.print(f"stderr: {e.stderr}")
            raise


def command_exists(command: str) -> bool:
    """Check if a command exists in the system."""
    return run_subprocess(f"which {command}", check=False).returncode == 0


def check_dependencies(verbose: bool):
    """Check and set up gcloud and kubectl dependencies.

    This function verifies that gcloud and kubectl are installed and properly configured.
    If kubectl is not installed, it attempts to install it using gcloud components.

    Args:
        verbose (bool): If True, provides more detailed output.

    Raises:
        EnvironmentError: If gcloud is not installed or kubectl cannot be configured.
    """
    if not command_exists("gcloud"):
        raise EnvironmentError("gcloud is not installed. Please install it first.")

    if not command_exists("kubectl"):
        console.print("kubectl is not installed. Installing it now...")
        run_subprocess("gcloud components install kubectl")

    # Authenticate gcloud
    active_account = (
        run_subprocess(
            "gcloud auth list --filter=status:ACTIVE --format=value'(ACCOUNT)'",
            capture_output=True,
            stream_output=verbose,
        )
        .stdout.strip()
        .split("\n")[0]
    )

    if not active_account:
        console.print("Authenticating gcloud...")
        run_subprocess("gcloud auth login", stream_output=verbose)

    # Configure kubectl
    project = "mtrx-hub-dev-3of"
    region = "us-central1"
    cluster = "compute-cluster"

    # Check if kubectl is already authenticated
    try:
        run_subprocess("kubectl get nodes", capture_output=True, stream_output=verbose)
        console.print("kubectl is already authenticated.")
    except subprocess.CalledProcessError:
        console.print("Authenticating kubectl...")
        run_subprocess(
            f"gcloud container clusters get-credentials {cluster} --project {project} --region {region}"
        )

    # Verify kubectl
    try:
        run_subprocess("kubectl get ns", capture_output=True, stream_output=verbose)
    except subprocess.CalledProcessError:
        raise EnvironmentError(
            "kubectl is not working. Please check your configuration."
        )


def build_push_docker(username: str, verbose: bool):
    """Build and push Docker image."""
    run_subprocess(f"make docker_push TAG={username}", stream_output=verbose)


def build_argo_template(run_name: str, release_version: str, username: str, namespace: str, pipeline_obj: Pipeline, is_test: bool, default_execution_resources: Optional[ArgoResourceConfig] = None) -> str:
    """Build Argo workflow template."""
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"

    matrix_root = Path(__file__).parent.parent.parent.parent
    metadata = bootstrap_project(matrix_root)
    package_name = metadata.package_name

    if is_test:
        release_folder_name = "tests"
    else:
        release_folder_name = "releases"

    return generate_argo_config(
        image=image_name,
        run_name=run_name,
        release_version=release_version,
        image_tag=run_name,
        namespace=namespace,
        username=username,
        package_name=package_name,
        release_folder_name=release_folder_name,
        pipeline=pipeline_obj,
        default_execution_resources=default_execution_resources,
    )

def save_argo_template(argo_template: str, template_directory: Path) -> str:
    file_path = template_directory / "argo-workflow-template.yml"
    with open(file_path, "w") as f:
        f.write(argo_template)
    return str(file_path)


def argo_template_lint(file_path: str, verbose: bool) -> str:
    run_subprocess(
        f"argo template lint {file_path}",
        check=True,
        stream_output=verbose,
    )

def ensure_namespace(namespace, verbose: bool):
    """Create or verify Kubernetes namespace."""
    result = run_subprocess(f"kubectl get namespace {namespace}", check=False)
    if result.returncode != 0:
        console.print(f"Namespace {namespace} does not exist. Creating it...")
        run_subprocess(f"kubectl create namespace {namespace}", check=True, stream_output=verbose)


def apply_argo_template(namespace, file_path: Path, verbose: bool):
    """Apply the Argo workflow template, making it available in the cluster.
    
    `kubectl apply -f <file_path> -n <namespace>` will make the template available as a resource (but will not create any other resources, and will not trigger the workshop).
    """
    cmd = f"kubectl apply -f {file_path} -n {namespace}"
    console.print(f"Running apply command: [blue]{cmd}[/blue]")
    run_subprocess(
        cmd,
        check=True,
        stream_output=verbose,
    )

def submit_workflow(run_name: str, namespace: str, verbose: bool):
    """Submit the Argo workflow and provide instructions for watching."""

    cmd = " ".join([
        "argo submit",
        f"--name {run_name}",
        f"-n {namespace}",
        f"--from wftmpl/{run_name}", # name of the template resource (created in previous step)
        f"-p run_name={run_name}",
        "-l submit-from-ui=false",
        "-o json"
    ])
    console.print(f"Running submit command: [blue]{cmd}[/blue]")
    result = run_subprocess(cmd, capture_output=True, stream_output=verbose)
    job_name = json.loads(result.stdout).get("metadata", {}).get("name")

    if not job_name:
        raise RuntimeError("Failed to retrieve job name from Argo submission.")

    console.print(f"Workflow submitted successfully with job name: {job_name}")
    console.print("\nTo watch the workflow progress, run the following command:")
    console.print(f"argo watch -n {namespace} {job_name}")
    console.print("\nTo view the workflow in the Argo UI, run:")
    console.print(f"argo get -n {namespace} {job_name}")

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
        run_name = run_subprocess(
            "git rev-parse --abbrev-ref HEAD", capture_output=True, stream_output=False
        ).stdout.strip()

    # Add a random suffix to the run_name
    random_sfx = str.lower(secrets.token_hex(4))
    unsanitized_name = f"{run_name}-{random_sfx}".rstrip("-")
    sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", unsanitized_name)
    return sanitized_name
