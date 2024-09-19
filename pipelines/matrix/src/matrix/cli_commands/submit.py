"""Command line tools for manipulating a Kedro project.

Intended to be invoked via `kedro`.
"""
# NOTE: This file was partially generated using AI assistance.

import json
import logging
import re
import secrets
import subprocess
import sys
from typing import Optional

import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")

from matrix.argo import _generate_argo_config

console = Console()


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


# fmt: off
@cli.command()
@click.option("--username", type=str, required=True, help="Specify the username to use")
@click.option("--namespace", type=str, default=None, help="Specify a custom namespace")
@click.option("--run-name", type=str, default=None, help="Specify a custom run name, defaults to branch")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose output")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
# fmt: on
def submit(username: str, namespace: str, run_name: str, verbose: bool, dry_run: bool):
    """Submit the end-to-end workflow."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_name = get_run_name(run_name)
    namespace = namespace or "argo-workflows"

    try:
        console.rule("[bold blue]Submitting Workflow")

        console.print("Checking dependencies...")
        check_dependencies(verbose=verbose)
        console.print("[green]✓[/green] Dependencies checked")

        console.print("Building and pushing Docker image...")
        build_push_docker(username, verbose=verbose)
        console.print("[green]✓[/green] Docker image built and pushed")

        console.print("Building Argo template...")
        build_argo_template(run_name, username, namespace, verbose=verbose)
        console.print("[green]✓[/green] Argo template built")

        console.print("Ensuring namespace...")
        ensure_namespace(namespace, verbose=verbose)
        console.print("[green]✓[/green] Namespace ensured")

        console.print("Applying Argo template...")
        apply_argo_template(namespace, verbose=verbose)
        console.print("[green]✓[/green] Argo template applied")

        if not dry_run:
            console.print("Submitting workflow...")
            submit_workflow(run_name, namespace, verbose=verbose)
            console.print("[green]✓[/green] Workflow submitted")

        console.print(Panel.fit(
            f"[bold green]Workflow {'prepared' if dry_run else 'submitted'} successfully![/bold green]\n"
            f"Run Name: {run_name}\n"
            f"Namespace: {namespace}",
            title="Submission Summary"
        ))

        prompt_browser_open(dry_run, f"https://argo.platform.dev.everycure.org/workflows/{namespace}/{run_name}")

    except Exception as e:
        console.print(f"[bold red]Error during submission:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def prompt_browser_open(dry_run: bool, workflow_url: str):
    """Prompts to open in a browser if not in dry run mode."""
    # if on macos:
    if sys.platform == "darwin":
        run_subprocess("say 'Would you like me to open this in a browser?'")
    if click.confirm("Would you like me to open this in a browser?"):
        click.launch(workflow_url)
    if sys.platform == "darwin":
        run_subprocess("say 'Certainly!'")
    console.print(f"[blue]Workflow URL: {workflow_url}[/blue]")


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


def build_argo_template(run_name, username, namespace, verbose: bool):
    """Build Argo workflow template."""
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
    _generate_argo_config(image_name, run_name, username, namespace, username)


def ensure_namespace(namespace, verbose: bool):
    """Create or verify Kubernetes namespace."""
    result = run_subprocess(f"kubectl get namespace {namespace}", check=False)
    if result.returncode != 0:
        console.print(f"Namespace {namespace} does not exist. Creating it...")
        run_subprocess(f"kubectl create namespace {namespace}", check=True, stream_output=verbose)


def apply_argo_template(namespace, verbose: bool):
    """Apply the Argo workflow template."""
    run_subprocess(
        f"kubectl apply -f templates/argo-workflow-template.yml -n {namespace}",
        check=True,
        stream_output=verbose,
    )


def submit_workflow(run_name, namespace, verbose: bool):
    """Submit the Argo workflow and provide instructions for watching."""
    submit_cmd = (
        f"argo submit --name {run_name} -n {namespace} "
        f"--from wftmpl/{run_name} -p run_name={run_name} "
        f"-l submit-from-ui=false --entrypoint __default__ -o json"
    )
    result = run_subprocess(submit_cmd, capture_output=True, stream_output=verbose)
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

    If a run_name is provided, it is returned as-is. Otherwise, a name is generated
    based on the current Git branch name with a random suffix.

    Args:
        run_name (Optional[str]): A custom run name provided by the user.

    Returns:
        str: The final run name to be used for the workflow.
    """
    if run_name:
        return run_name
    branch_name = run_subprocess(
        "git rev-parse --abbrev-ref HEAD", capture_output=True, stream_output=False
    ).stdout.strip()
    branch_sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", branch_name).rstrip("-")
    random_sfx = str.lower(secrets.token_hex(4))
    return f"{branch_sanitized}-{random_sfx}"
