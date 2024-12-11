import json
import logging
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union

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
@click.option("--release", type=str, default=None, required=False, help="Specify a custom release name")
@click.option("--pipeline", "-p", type=str, default="modelling_run", help="Specify which pipeline to execute")
@click.option("--verbose", "-v", is_flag=True, default=True, help="Enable verbose output")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
@click.option("--from-nodes", type=str, default="", help="Specify nodes to run from", callback=split_string)
# fmt: on
def submit(username: str, namespace: str, run_name: Optional[str], release: Optional[str], pipeline: str, verbose: bool, dry_run: bool, from_nodes: List[str]):
    """Submit the end-to-end workflow. """
    if verbose:
        log.setLevel(logging.DEBUG)

    if pipeline in {"fabricator", "test"}:
        raise ValueError("Submitting test pipeline to Argo will result in overwriting source data")
    
    if from_nodes:
        if not click.confirm("Using 'from-nodes' is highly experimental and may break due to MLFlow issues with tracking the right run. Are you sure you want to continue?", default=False):
            raise click.Abort()

    if release and release_exists(release):
        raise ValueError("The specified release already exists. You do not want to overwrite it.")

    try:
        pipeline_obj = kedro_pipelines[pipeline]
    except KeyError:
        raise ValueError("Pipeline requested for execution not found")
    # As a temporary measure, we pass both pipeline for execution and list of pipelines. In the future, we will merge the two.
    if from_nodes:
        pipeline_obj = pipeline_obj.from_nodes(*from_nodes)

    run_name = get_run_name(run_name)
    pipeline_obj.name = pipeline

    summarize_submission(run_name, namespace, pipeline, release_version=release)
    _submit(
        username=username,
        namespace=namespace,
        run_name=run_name,
        release_version=release,
        pipeline_obj=pipeline_obj,
        verbose=verbose,
        dry_run=dry_run,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
    )


def _submit(
        username: str, 
        namespace: str, 
        run_name: str, 
        release_version: Optional[str],
        pipeline_obj: Pipeline,
        verbose: bool, 
        dry_run: bool, 
        template_directory: Path,
        allow_interactions: bool = True,
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
    """
    
    try:
        console.rule("[bold blue]Submitting Workflow")

        check_dependencies(verbose=verbose)

        argo_template = build_argo_template(
            run_name=run_name,
            release_version=release_version,
            username=username,
            namespace=namespace,
            pipeline_obj=pipeline_obj
        )

        file_path = save_argo_template(argo_template, template_directory)

        argo_template_lint(file_path, verbose=verbose)

        if dry_run:
            return

        build_push_docker(run_name, verbose=verbose)

        ensure_namespace(namespace, verbose=verbose)

        apply_argo_template(namespace, file_path, verbose=verbose)

        submit_workflow(run_name, namespace, verbose=verbose)

        console.print(Panel.fit(
            f"[bold green]Workflow submitted successfully![/bold green]\n"
            f"Run Name: {run_name}\n"
            f"Namespace: {namespace}",
            title="Submission Summary"
        ))

        if allow_interactions and click.confirm("Do you want to open the workflow in your browser?", default=False):
            workflow_url = f"https://argo.platform.dev.everycure.org/workflows/{namespace}/{run_name}"
            click.launch(workflow_url)
            console.print(f"[blue]Opened workflow in browser: {workflow_url}[/blue]")

    except Exception as e:
        console.print(f"[bold red]Error during submission:[/bold red] {str(e)}")
        console.print_exception()
        sys.exit(1)


def summarize_submission(run_name: str, namespace: str, pipeline: str, release_version: Optional[str]):
    summary = (
        "[bold green]About to submit workflow:[/bold green]",
        f"Run Name: {run_name}"
        f"Namespace: {namespace}"
        f"Pipeline: {pipeline}"
        f"Data Release Version: {release_version}" if release_version else "Output to tests: true",
    )
    console.print(Panel.fit("\n".join(summary), title="Submission Summary"))
    console.print("\nReminder: A data release should only be submitted once and not overwritten.\n"
                  "If you need to make changes, please make this part of the next release.\n"
                  "Experiments (modelling pipeline) are nested under the release and can be overwritten.\n\n")

    if not click.confirm("Are you sure you want to submit the workflow?", default=False):
        raise click.Abort()
        

def run_subprocess(
    cmd: str,
    check: bool = True,
    shell: bool = True,
    stream_output: bool = True,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and handle errors.

    Args:
        cmd: Command string to execute
        check: If True, raise CalledProcessError on non-zero exit status
        shell: If True, execute the command through the shell
        stream_output: If True, capture and stream output to stdout/stderr.
                      If False, send output directly to system stdout/stderr.
        cwd: If provided, change the working directory for the execution of this command
    Returns:
        CompletedProcess instance with stdout/stderr (if stream_output=True)
    """
    process = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE if stream_output else None,
        stderr=subprocess.PIPE if stream_output else None,
        text=True,
        bufsize=1,
        cwd=cwd,
    )

    stdout, stderr = [], []

    if stream_output:
        while True:
            out_line = process.stdout.readline() if process.stdout else ''
            err_line = process.stderr.readline() if process.stderr else ''

            if not out_line and not err_line and process.poll() is not None:
                break

            if out_line:
                sys.stdout.write(out_line)
                sys.stdout.flush()
                stdout.append(out_line)
            if err_line:
                sys.stderr.write(err_line)
                sys.stderr.flush()
                stderr.append(err_line)

        # Get any remaining output
        out, err = process.communicate()
        if out:
            stdout.append(out)
        if err:
            stderr.append(err)
    else:
        process.wait()

    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd,
            ''.join(stdout) if stdout else None,
            ''.join(stderr) if stderr else None
        )

    return subprocess.CompletedProcess(
        cmd, process.returncode,
        ''.join(stdout) if stdout else None,
        ''.join(stderr) if stderr else None
    )

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
    console.print("Checking dependencies...")

    if not command_exists("gcloud"):
        raise EnvironmentError("gcloud is not installed. Please install it first.")

    if not command_exists("kubectl"):
        console.print("kubectl is not installed. Installing it now...")
        run_subprocess("gcloud components install kubectl")

    # Authenticate gcloud
    active_account = (
        run_subprocess(
            "gcloud auth list --filter=status:ACTIVE --format=value'(ACCOUNT)'",
            stream_output=verbose,
        )
        .stdout.strip()
        .split("\n")[0]
    )

    if not active_account:
        console.print("Authenticating gcloud...")
        run_subprocess("gcloud auth login", stream_output=verbose)

    # Check if kubectl is already authenticated
    try:
        run_subprocess("kubectl get nodes", stream_output=verbose)
        console.print("[green]✓[/green] kubectl authenticated")
    except subprocess.CalledProcessError:
        console.print("Authenticating kubectl...")
        run_subprocess(
            f"gcloud container clusters get-credentials {os.environ['GCP_CLUSTER_NAME']} --project {os.environ['GCP_PROJECT_ID']} --region {os.environ['GCP_MAIN_REGION']}",
            stream_output=verbose,
        )
        console.print("[green]✓[/green] kubectl authenticated")

    # Verify kubectl
    try:
        run_subprocess("kubectl get ns", stream_output=verbose)
    except subprocess.CalledProcessError:
        raise EnvironmentError(
            "kubectl is not working. Please check your configuration."
        )
    console.print("[green]✓[/green] Dependencies checked")



def build_push_docker(tag: str, verbose: bool):
    """Build and push Docker image."""
    console.print("Building Docker image...")
    run_subprocess(f"make docker_push TAG={tag}", stream_output=verbose)
    console.print("[green]✓[/green] Docker image built and pushed")


def build_argo_template(run_name: str, username: str, namespace: str, pipeline_obj: Pipeline, release_version: Optional[str]=None, default_execution_resources: Optional[ArgoResourceConfig] = None) -> str:
    """Build Argo workflow template."""
    matrix_root = Path(__file__).parents[3]
    metadata = bootstrap_project(matrix_root)
    package_name = metadata.package_name

    release_folder_name = "releases" if release_version else "tests"

    console.print("Building Argo template...")
    generated_template = generate_argo_config(
        image=os.environ["MATRIX_IMAGE"],
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
    console.print("[green]✓[/green] Argo template built")

    return generated_template


def save_argo_template(argo_template: str, template_directory: Path) -> str:
    console.print("Writing Argo template...")
    file_path = template_directory / "argo-workflow-template.yml"
    file_path.write_text(argo_template)
    console.print(f"[green]✓[/green] Argo template saved to {file_path}")
    return str(file_path)


def argo_template_lint(file_path: str, verbose: bool) -> str:
    console.print("Linting Argo template...")
    run_subprocess(
        f"argo template lint {file_path}",
        check=True,
        stream_output=verbose,
    )
    console.print("[green]✓[/green] Argo template linted")

def ensure_namespace(namespace, verbose: bool):
    """Create or verify Kubernetes namespace."""
    console.print("Ensuring Kubernetes namespace...")
    result = run_subprocess(f"kubectl get namespace {namespace}", check=False)
    console.print("[green]✓[/green] Namespace ensured")
    if result.returncode != 0:
        console.print(f"Namespace {namespace} does not exist. Creating it...")
        run_subprocess(f"kubectl create namespace {namespace}", check=True, stream_output=verbose)


def apply_argo_template(namespace, file_path: Union[Path, str], verbose: bool):
    """Apply the Argo workflow template, making it available in the cluster.
    
    `kubectl apply -f <file_path> -n <namespace>` will make the template available as a resource (but will not create any other resources, and will not trigger the workshop).
    """
    console.print("Applying Argo template...")

    cmd = f"kubectl apply -f {file_path} -n {namespace}"
    console.print(f"Running apply command: [blue]{cmd}[/blue]")
    run_subprocess(
        cmd,
        check=True,
        stream_output=verbose,
    )
    console.print("[green]✓[/green] Argo template applied")

def submit_workflow(run_name: str, namespace: str, verbose: bool):
    """Submit the Argo workflow and provide instructions for watching."""
    console.print("Submitting workflow for pipeline...")

    cmd = " ".join([
        "argo submit",
        f"--name {run_name}",
        f"--namespace {namespace}",
        f"--from wftmpl/{run_name}", # name of the template resource (created in previous step)
        f"--parameter run_name={run_name}",
        "--labels submit-from-ui=false",
        "--output json"
    ])
    console.print(f"Running submit command: [blue]{cmd}[/blue]")
    result = run_subprocess(cmd, stream_output=verbose)
    job_name = json.loads(result.stdout).get("metadata", {}).get("name")

    if not job_name:
        raise RuntimeError("Failed to retrieve job name from Argo submission.")

    console.print(f"Workflow submitted successfully with job name: {job_name}")
    console.print("\nTo watch the workflow progress, run the following command:")
    console.print(f"argo watch -n {namespace} {job_name}")
    console.print("\nTo view the workflow in the Argo UI, run:")
    console.print(f"argo get -n {namespace} {job_name}")
    console.print("[green]✓[/green] Workflow submitted")


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
            "git rev-parse --abbrev-ref HEAD",
            stream_output=True,
            cwd=Path(__file__).parent.absolute(),
        ).stdout.strip()

    # Add a random suffix to the run_name
    random_sfx = secrets.token_hex(4).lower()
    unsanitized_name = f"{run_name}-{random_sfx}"
    sanitized_name = re.sub(r"[^a-zA-Z0-9-]", "-", unsanitized_name)
    return sanitized_name

def release_exists(name: str) -> bool:
    assert '.' in name, f"{name} doesn't look like a version number"
    remote_tags = run_subprocess(cmd="git ls-remote --tags --quiet", stream_output=False,
                                 cwd=Path(__file__).parent.absolute() # git cmd won't work well when outside of a git repo
                                 ).stdout
    return name in remote_tags
