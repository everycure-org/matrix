"""Command line tools for manipulating a Kedro project.

Intended to be invoked via `kedro`.
"""

import json
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set

import click
from kedro.framework.cli.project import (
    ASYNC_ARG_HELP,
    CONF_SOURCE_HELP,
    CONFIG_FILE_HELP,
    FROM_INPUTS_HELP,
    FROM_NODES_HELP,
    LOAD_VERSION_HELP,
    NODE_ARG_HELP,
    PARAMS_ARG_HELP,
    PIPELINE_ARG_HELP,
    RUNNER_ARG_HELP,
    TAG_ARG_HELP,
    TO_NODES_HELP,
    TO_OUTPUTS_HELP,
    project_group,
)
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    _config_file_callback,
    _split_load_versions,
    _split_params,
    env_option,
    split_node_names,
    split_string,
)
import secrets
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.framework.project import pipelines, settings
from kedro.io import DataCatalog
from kedro.pipeline.pipeline import Pipeline
from kedro.utils import load_obj
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from matrix.argo import _generate_argo_config
from matrix.session import KedroSessionWithFromCatalog
from matrix.utils.submit_cli import *

console = Console()


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


# fmt: off
@project_group.command()
@click.option("--username", type=str, required=True, help="Specify the username to use")
@click.option( "--namespace", type=str, default=None, help="Specify a custom namespace")
@click.option("--run-name", type=str, default=None, help="Specify a custom run name")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose output")
# fmt: on
def submit(username, namespace, run_name, verbose):
    """Submit the end-to-end workflow."""
    run_name = get_run_name(run_name)
    namespace = namespace or f"dev-{username}"
    try:
        console.rule("[bold blue]Submitting Workflow")
        check_dependencies(verbose=verbose)
        console.print("[green]✓[/green] Dependencies checked")
        build_push_docker(username, verbose=verbose)
        console.print("[green]✓[/green] Docker image built and pushed")
        build_argo_template(run_name, username, namespace, verbose=verbose)
        console.print("[green]✓[/green] Argo template built")
        ensure_namespace(namespace, verbose=verbose)
        console.print("[green]✓[/green] Namespace ensured")
        apply_argo_template(namespace, verbose=verbose)
        console.print("[green]✓[/green] Argo template applied")
        submit_workflow(run_name, namespace, verbose=verbose)
        console.print("[green]✓[/green] Workflow submitted")
        console.print(Panel.fit(
            f"[bold green]Workflow submitted successfully![/bold green]\n"
            f"Run Name: {run_name}\n"
            f"Namespace: {namespace}",
            title="Submission Summary"
        ))
        
        # New code to prompt user about opening the workflow in browser
        if click.confirm("Do you want to open the workflow in your browser?"):
            workflow_url = f"https://argo.platform.dev.everycure.org/workflows/{namespace}/{run_name}"
            click.launch(workflow_url)
            console.print(f"[blue]Opened workflow in browser: {workflow_url}[/blue]")
    except Exception as e:
        console.print(Panel(f"[bold red]Error during submission:[/bold red]\n{str(e)}", 
                            title="Error", border_style="red"))
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
            click.echo(f"Error executing command: {cmd}")
            click.echo(f"Exit code: {e.returncode}")
            if e.stdout:
                click.echo(f"stdout: {e.stdout}")
            if e.stderr:
                click.echo(f"stderr: {e.stderr}")
            raise


def command_exists(command: str) -> bool:
    """Check if a command exists in the system."""
    return run_subprocess(f"which {command}", check=False).returncode == 0


def check_dependencies(verbose: bool):
    """Check and set up gcloud and kubectl."""
    if not command_exists("gcloud"):
        raise EnvironmentError("gcloud is not installed. Please install it first.")

    if not command_exists("kubectl"):
        click.echo("kubectl is not installed. Installing it now...")
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
        click.echo("Authenticating gcloud...")
        run_subprocess("gcloud auth login", stream_output=verbose)

    # Configure kubectl
    project = "mtrx-hub-dev-3of"
    region = "us-central1"
    cluster = "compute-cluster"

    # Check if kubectl is already authenticated
    try:
        run_subprocess("kubectl get nodes", capture_output=True, stream_output=verbose)
        click.echo("kubectl is already authenticated.")
    except subprocess.CalledProcessError:
        click.echo("Authenticating kubectl...")
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
    click.echo("Building and pushing Docker image...")
    image_name = f"us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix:{username}"
    run_subprocess(f"make docker_push TAG={username}", stream_output=verbose)
    click.echo(f"Successfully built and pushed Docker image: {image_name}")


def build_argo_template(run_name, username, namespace, verbose: bool):
    """Build Argo workflow template."""
    click.echo("Building Argo workflow template...")
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    _generate_argo_config(image_name, run_name, username, namespace)


def ensure_namespace(namespace, verbose: bool):
    """Create or verify Kubernetes namespace."""
    click.echo(f"Using namespace: {namespace}")
    result = run_subprocess(f"kubectl get namespace {namespace}", check=False)
    if result.returncode != 0:
        click.echo(f"Namespace {namespace} does not exist. Creating it...")
        run_subprocess(f"kubectl create namespace {namespace}", check=True, stream_output=verbose)


def apply_argo_template(namespace, verbose: bool):
    """Apply the Argo workflow template."""
    click.echo("Applying Argo workflow template...")
    run_subprocess(
        f"kubectl apply -f templates/argo-workflow-template.yml -n {namespace}",
        check=True,
        stream_output=verbose,
    )


def submit_workflow(run_name, namespace, verbose: bool):
    """Submit the Argo workflow and provide instructions for watching."""
    click.echo("Submitting Argo workflow...")
    submit_cmd = (
        f"argo submit --name {run_name} -n {namespace} "
        f"--from wftmpl/{run_name} -p run_name={run_name} "
        f"-l submit-from-ui=false --entrypoint __default__ -o json"
    )
    result = run_subprocess(submit_cmd, capture_output=True, stream_output=verbose)
    job_name = json.loads(result.stdout).get("metadata", {}).get("name")

    if not job_name:
        raise RuntimeError("Failed to retrieve job name from Argo submission.")

    click.echo(f"Workflow submitted successfully with job name: {job_name}")
    click.echo("\nTo watch the workflow progress, run the following command:")
    click.echo(f"argo watch -n {namespace} {job_name}")
    click.echo("\nTo view the workflow in the Argo UI, run:")
    click.echo(f"argo get -n {namespace} {job_name}")

def get_run_name(run_name: Optional[str]) -> str:
    """Get the experiment name based on input or Git branch."""
    if run_name:
        return run_name
    branch_name = run_subprocess(
        "git rev-parse --abbrev-ref HEAD", capture_output=True
    ).stdout.strip()
    branch_sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", branch_name).rstrip("-")
    random_sfx = str.lower(secrets.token_hex(4))
    return f"{branch_sanitized}-{random_sfx}"
