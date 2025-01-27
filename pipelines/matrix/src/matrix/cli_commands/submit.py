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
from matrix.git_utils import (
    BRANCH_NAME_REGEX,
    git_tag_exists,
    has_dirty_git,
    has_legal_branch_name,
    has_unpushed_commits,
)
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
@click.option("--pipeline", "-p", type=str, default="modelling_run", help="Specify which pipeline to execute")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Disable verbose output")
@click.option("--dry-run", "-d", is_flag=True, default=False, help="Does everything except submit the workflow")
@click.option("--from-nodes", type=str, default="", help="Specify nodes to run from", callback=split_string)
@click.option("--is-test", is_flag=True, default=False, help="Submit to test folder")
@click.option("--headless", is_flag=True, default=False, help="Skip confirmation prompt")
@click.option("--environment", "-e", type=str, default="cloud", help="Kedro environment to execute in")
# fmt: on
def submit(
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
    environment: str
):
    """Submit the end-to-end workflow. """
    if not quiet:
        log.setLevel(logging.DEBUG)

    if pipeline in ('data_release', 'kg_release'):
        abort_if_unmet_git_requirements(release_version)

    if pipeline not in kedro_pipelines.keys():
        raise ValueError("Pipeline requested for execution not found")
    
    if pipeline in ["fabricator", "test"]:
        raise ValueError("Submitting test pipeline to Argo will result in overwriting source data")
    
    if not headless and from_nodes:
        if not click.confirm("Using 'from-nodes' is highly experimental and may break due to MLFlow issues with tracking the right run. Are you sure you want to continue?", default=False):
            raise click.Abort()

    pipeline_obj = kedro_pipelines[pipeline]
    if from_nodes:
        pipeline_obj = pipeline_obj.from_nodes(*from_nodes)

    run_name = get_run_name(run_name)
    pipeline_obj.name = pipeline


    if not dry_run:
        summarize_submission(run_name, namespace, pipeline, environment, is_test, release_version, headless)
   
    _submit(
        username=username,
        namespace=namespace,
        run_name=run_name,
        release_version=release_version,
        pipeline_obj=pipeline_obj,
        verbose=not quiet,
        dry_run=dry_run,
        template_directory=ARGO_TEMPLATES_DIR_PATH,
        allow_interactions=not headless,
        is_test=is_test,
        environment=environment,
    )


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
        allow_interactions (bool): If True, allow prompts for confirmation.
        is_test (bool): If True, submit to test folder, not release folder.
    """
    
    try:
        console.rule("[bold blue]Submitting Workflow")

        if not can_talk_to_kubernetes():
            raise EnvironmentError("Cannot communicate with Kubernetes")

        argo_template = build_argo_template(run_name, release_version, username, namespace, pipeline_obj, environment, is_test=is_test, )

        file_path = save_argo_template(argo_template, template_directory)

        argo_template_lint(file_path, verbose=verbose)

        if dry_run:
            return

        build_push_docker(run_name, verbose=True)

        ensure_namespace(namespace, verbose=verbose)

        apply_argo_template(namespace, file_path, verbose=verbose)

        submit_workflow(run_name, namespace, verbose=False)

        console.print(Panel.fit(
            f"[bold green]Workflow {'prepared' if dry_run else 'submitted'} successfully![/bold green]\n"
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
        if verbose:
            console.print_exception()
        sys.exit(1)



def summarize_submission(run_name: str, namespace: str, pipeline: str, environment: str, is_test: bool, release_version: str, headless:bool):
    console.print(Panel.fit(
        f"[bold green]About to submit workflow:[/bold green]\n"
        f"Run Name: {run_name}\n"
        f"Namespace: {namespace}\n"
        f"Pipeline: {pipeline}\n"
        f"Environment: {environment}\n"
        f"Writing to test folder: {is_test}\n"
        f"Data Release Version: {release_version}\n",
        title="Submission Summary"
    ))
    console.print("Reminder: A data release should only be submitted once and not overwritten.\n"
                  "If you need to make changes, please make this part of the next release.\n"
                  "Experiments (modelling pipeline) are nested under the release and can be overwritten.\n\n")
    
    if not headless:
        if not click.confirm("Are you sure you want to submit the workflow?", default=False):
            raise click.Abort()
        

def run_subprocess(
    cmd: str,
    check: bool = True,
    shell: bool = True,
    stream_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and handle errors.

    Args:
        cmd: Command string to execute
        check: If True, raise CalledProcessError on non-zero exit status
        shell: If True, execute the command through the shell
        stream_output: If True, capture and stream output to stdout/stderr.
                      If False, send output directly to system stdout/stderr.
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


def can_talk_to_kubernetes(
    project: str = "mtrx-hub-dev-3of",
    region:  str = "us-central1",
    cluster_name: str = "compute-cluster",
) -> bool:
    """Check if one can communicate with the Kubernetes cluster, using the kubectl CLI.

    If kubectl is not installed, it attempts to install and configure it using gcloud components.

    Raises:
        EnvironmentError: If gcloud is not installed or kubectl cannot be configured.
    """

    def run_gcloud_cmd(s: str, timeout: int = 300) -> None:
        try:
            subprocess.check_output(s, shell=True, stderr=subprocess.PIPE, timeout=timeout)
        except FileNotFoundError as e:
            raise EnvironmentError("gcloud is not installed. Please install it first.") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"The command '{s}' took more than {timeout}s to complete.") from e
        except subprocess.CalledProcessError as e:
            if b"You do not currently have an active account selected" in e.stderr:
                log.warning(
                    "You're not using an authenticated account to interact with the gcloud CLI. Attempting to log you in…")
                run_gcloud_cmd("gcloud auth login")
                log.info("Logged in to GCS.")
                subprocess.check_output(s, shell=True, stderr=subprocess.PIPE, timeout=timeout)
            else:
                pretty_report_on_error(e)

    def refresh_kube_credentials() -> None:
        log.debug("Refreshing kubectl credentials…")
        run_gcloud_cmd(
            f"gcloud container clusters get-credentials {cluster_name} --project {project} --region {region}")

    def get_kubernetes_context() -> str:
        return subprocess.check_output(["kubectl", "config", "current-context"], text=True).strip()

    def use_kubernetes_context(context: str) -> subprocess.CompletedProcess[bytes]:
        log.info(f"Switching kubernetes context to '{context}'")
        return subprocess.run(["kubectl", "config", "use-context", context], check=True, stdout=subprocess.DEVNULL)

    def pretty_report_on_error(e: subprocess.CalledProcessError):
        try:
            raise EnvironmentError(f"Calling '{e.cmd}' failed, with stderr: '{e.stderr}'") from e
        except EnvironmentError:
            console.print_exception()
            raise

    right_kube_context = "_".join(("gke", project, region, cluster_name))
    try:
        current_context = get_kubernetes_context()
    except FileNotFoundError:
        log.warning("kubectl is not installed. Attempting to install it now…")
        run_gcloud_cmd("gcloud components install kubectl")
        current_context = get_kubernetes_context()

    if current_context != right_kube_context:
        log.debug(f"Current context ({current_context}) does not match intended ({right_kube_context}).")
        use_kubernetes_context(right_kube_context)

    test_cmd = "kubectl get nodes"
    # Drop the stdout of the test_cmd, but track any errors, so they can be logged
    try:
        subprocess.run(test_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        log.debug(f"'{test_cmd}' failed. Reason: {e.stderr}")
        if b"Unauthorized" in e.stderr:
            refresh_kube_credentials()
            subprocess.run(test_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        else:
            pretty_report_on_error(e)

    console.print("[green]✓[/green] kubectl authenticated")
    return True


def build_push_docker(username: str, verbose: bool):
    """Build and push Docker image."""
    console.print("Building Docker image...")
    run_subprocess(f"make docker_push TAG={username}", stream_output=False)
    console.print("[green]✓[/green] Docker image built and pushed")


def build_argo_template(
    run_name: str, 
    release_version: str, 
    username: str, 
    namespace: str, 
    pipeline_obj: Pipeline, 
    environment: str,
    is_test: bool, 
    default_execution_resources: Optional[ArgoResourceConfig] = None
) -> str:
    """Build Argo workflow template."""
    image_name = "us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix"
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
        image_tag=run_name,
        namespace=namespace,
        username=username,
        package_name=package_name,
        release_folder_name=release_folder_name,
        pipeline=pipeline_obj,
        environment=environment,
        default_execution_resources=default_execution_resources,
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


def apply_argo_template(namespace, file_path: Path, verbose: bool):
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
        f"-n {namespace}",
        f"--from wftmpl/{run_name}", # name of the template resource (created in previous step)
        f"-p run_name={run_name}",
        "-l submit-from-ui=false",
        "-o json"
    ])
    console.print(f"Running submit command: [blue]{cmd}[/blue]")
    console.print(f"\nSee your workflow in the ArgoCD UI here: [blue]https://argo.platform.dev.everycure.org/workflows/argo-workflows/{run_name}[/blue]")
    result = run_subprocess(cmd)
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
            "git rev-parse --abbrev-ref HEAD", stream_output=True
        ).stdout.strip()

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

    if has_dirty_git():
        errors.append("Repository has uncommitted changes or untracked files.")

    if not has_legal_branch_name():
        errors.append(f"Your branch name doesn't match the regex: {BRANCH_NAME_REGEX}")

    if has_unpushed_commits():
        errors.append(f"You have commits not pushed to remote.")

    if git_tag_exists(release_version):
        errors.append(f"The git tag for the release version you specified already exists.")

    if errors:
        error_list = "\n".join(errors)
        raise RuntimeError(f"Submission failed due to the following issues:\n\n{error_list}")
