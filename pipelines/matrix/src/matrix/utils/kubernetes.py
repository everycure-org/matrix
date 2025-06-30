import logging
import subprocess
from os import environ
from pathlib import Path

import requests

from matrix.utils.system import run_subprocess

logger = logging.getLogger(__name__)


def can_talk_to_kubernetes(
    region: str = "us-central1",
    cluster_name: str = "compute-cluster",
) -> bool:
    """Check if one can communicate with the Kubernetes cluster, using the kubectl CLI.

    If kubectl is not installed, it attempts to install and configure it using gcloud components.

    Args:
        project: GCP project ID. If None, will auto-detect from gcloud config.
        region: GCP region where the cluster is located.
        cluster_name: Name of the GKE cluster.

    Raises:
        EnvironmentError: If gcloud is not installed or kubectl cannot be configured.
    """

    project = get_gcp_project_from_config()
    logger.info(f"Auto-detected GCP project: {project}")

    def add_impersonation_flag(s: str) -> str:
        """Add impersonation flag to gcloud command if SPARK_IMPERSONATION_SERVICE_ACCOUNT is set."""
        sa = environ.get("SPARK_IMPERSONATION_SERVICE_ACCOUNT")
        if sa and "gcloud" in s and "--impersonate-service-account" not in s:
            return s.replace("gcloud", f"gcloud --impersonate-service-account={sa}")
        return s

    def run_gcloud_cmd(s: str, timeout: int = 300) -> None:
        s = add_impersonation_flag(s)

        try:
            subprocess.check_output(s, shell=True, stderr=subprocess.PIPE, timeout=timeout)
        except FileNotFoundError as e:
            raise EnvironmentError("gcloud is not installed. Please install it first.") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"The command '{s}' took more than {timeout}s to complete.") from e
        except subprocess.CalledProcessError as e:
            if b"You do not currently have an active account selected" in e.stderr:
                logger.warning(
                    "You're not using an authenticated account to interact with the gcloud CLI. Attempting to log you in…"
                )
                run_gcloud_cmd("gcloud auth login")
                logger.info("Logged in to GCS.")
                subprocess.check_output(s, shell=True, stderr=subprocess.PIPE, timeout=timeout)
            else:
                pretty_report_on_error(e)

    def refresh_kube_credentials() -> None:
        """Refresh kubectl credentials for the specified GKE cluster."""
        # We do not want to refresh credentials if running in GitHub Actions, as it is not needed there.
        # In GitHub Actions, the credentials are already set up by the action.
        if not environ.get("GITHUB_ACTIONS"):
            logger.debug("Refreshing kubectl credentials…")
            refresh_command = (
                f"gcloud container clusters get-credentials {cluster_name} --project {project} --region {region}"
            )
            run_gcloud_cmd(refresh_command)

    def get_kubernetes_context() -> str:
        return subprocess.check_output(["kubectl", "config", "current-context"], text=True).strip()

    def use_kubernetes_context(context: str) -> subprocess.CompletedProcess[bytes]:
        logger.info(f"Switching kubernetes context to '{context}'")
        return subprocess.run(["kubectl", "config", "use-context", context], check=True, stdout=subprocess.DEVNULL)

    def pretty_report_on_error(e: subprocess.CalledProcessError):
        try:
            raise EnvironmentError(f"Calling '{e.cmd}' failed, with stderr: '{e.stderr}'") from e
        except EnvironmentError:
            raise

    # Refresh credentials before running the test command.
    refresh_kube_credentials()

    right_kube_context = "_".join(("gke", project, region, cluster_name))
    try:
        current_context = get_kubernetes_context()
    except FileNotFoundError:
        logger.warning("kubectl is not installed. Attempting to install it now…")
        run_gcloud_cmd("gcloud components install kubectl")
        current_context = get_kubernetes_context()

    if current_context != right_kube_context:
        logger.debug(f"Current context ({current_context}) does not match intended ({right_kube_context}).")
        use_kubernetes_context(right_kube_context)

    test_cmd = "kubectl get nodes"
    # Drop the stdout of the test_cmd, but track any errors, so they can be logged
    try:
        subprocess.run(test_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logger.debug(f"'{test_cmd}' failed. Reason: {e.stderr}")
        pretty_report_on_error(e)

    return True


def namespace_exists(namespace: str):
    """Function to verify whether kubernetes namespace exists."""
    result = run_subprocess(f"kubectl get namespace {namespace}", check=False)
    return result.returncode == 0


def create_namespace(namespace: str, verbose: bool):
    """Function to create a kubernetes namespace."""
    run_subprocess(f"kubectl create namespace {namespace}", check=True, stream_output=verbose)


def apply(namespace, file_path: Path, verbose: bool):
    """Apply file to kubernetes namespace

    `kubectl apply -f <file_path> -n <namespace>` will make the template available as a resource (but will not create any other resources, and will not trigger the workshop).
    """

    cmd = f"kubectl apply -f {file_path} -n {namespace}"
    run_subprocess(
        cmd,
        check=True,
        stream_output=verbose,
    )


def get_gcp_project_from_metadata() -> str:
    """Get GCP project ID from metadata server (works in GKE/GCP environments).

    This function tries to get the project ID from the GCP metadata server,
    which is available to pods running in GKE or GCP environments.

    Returns:
        str: The GCP project ID

    Raises:
        RuntimeError: If metadata server is not accessible or returns invalid data
    """
    try:
        # GCP metadata server endpoint
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
        headers = {"Metadata-Flavor": "Google"}

        response = requests.get(metadata_url, headers=headers, timeout=5)
        response.raise_for_status()

        project_id = response.text.strip()
        if project_id:
            logger.debug(f"Found GCP project from metadata server: {project_id}")
            return project_id
        else:
            raise RuntimeError("Empty project ID returned from metadata server")

    except Exception as e:
        raise RuntimeError(f"Failed to get project ID from GCP metadata server: {e}") from e


def get_gcp_project_from_config() -> str:
    """Get the active GCP project ID from gcloud configuration.

    This function fetches the project ID from the currently active gcloud configuration
    profile, which is controlled by the same authentication that can_talk_to_kubernetes
    function uses.

    Returns:
        str: The active GCP project ID

    Raises:
        EnvironmentError: If gcloud is not installed or no project is configured
        RuntimeError: If unable to determine the project ID
    """

    def add_impersonation_flag(cmd: str) -> str:
        """Add impersonation flag to gcloud command if SPARK_IMPERSONATION_SERVICE_ACCOUNT is set."""
        sa = environ.get("SPARK_IMPERSONATION_SERVICE_ACCOUNT")
        if sa and "gcloud" in cmd and "--impersonate-service-account" not in cmd:
            return cmd.replace("gcloud", f"gcloud --impersonate-service-account={sa}")
        return cmd

    try:
        # Get project ID from gcloud config
        cmd = "gcloud config get-value project"
        cmd = add_impersonation_flag(cmd)

        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.PIPE, text=True).strip()

        if not result or result == "(unset)":
            raise RuntimeError(
                "No active GCP project found in gcloud configuration. "
                "Please set one using: gcloud config set project PROJECT_ID"
            )

        logger.debug(f"Found active GCP project: {result}")
        return result

    except FileNotFoundError as e:
        raise EnvironmentError("gcloud is not installed. Please install it first.") from e
    except subprocess.CalledProcessError as e:
        # Handle both string and bytes stderr
        if e.stderr:
            error_msg = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
        else:
            error_msg = "Unknown error"
        raise RuntimeError(f"Failed to get GCP project from gcloud config: {error_msg}") from e


def get_runtime_gcp_project_id() -> str:
    """Get the runtime GCP project ID using cloud-native detection methods.

    This function tries multiple approaches to determine the GCP project ID:
    1. GCP metadata server (for GKE/GCP environments)
    2. gcloud CLI configuration (for local development)

    Returns:
        str: The GCP project ID to use for runtime operations

    Raises:
        RuntimeError: If no project ID can be determined from any source
    """
    # Try GCP metadata server first (works in GKE/GCP environments)
    try:
        project = get_gcp_project_from_metadata()
        logger.info(f"Using project ID from GCP metadata server: {project}")
        return project
    except RuntimeError as metadata_error:
        logger.debug(f"GCP metadata server unavailable: {metadata_error}")

    # Try gcloud CLI configuration (works for local development)
    try:
        project = get_gcp_project_from_config()
        logger.info(f"Using project ID from gcloud config: {project}")
        return project
    except (EnvironmentError, RuntimeError) as gcloud_error:
        logger.debug(f"gcloud config unavailable: {gcloud_error}")

    # If all methods fail, raise an error
    raise RuntimeError(
        "Could not determine GCP project ID from any source. Tried: "
        "1) GCP metadata server, 2) gcloud config. "
        "Please ensure the environment is properly configured with either GKE Workload Identity or gcloud authentication."
    )


def get_runtime_mlflow_url(project_id: str = None) -> str:
    """Get the runtime MLflow URL, with smart defaults based on project.

    Args:
        project_id: GCP project ID to use for determining environment

    Returns:
        str: The MLflow URL to use
    """
    # Check if explicitly set in environment
    env_url = environ.get("MLFLOW_URL")
    if env_url:
        logger.debug(f"Using MLflow URL from environment: {env_url}")
        return env_url

    # Auto-detect based on project ID
    if project_id is None:
        project_id = get_runtime_gcp_project_id()

    # Determine environment from project ID
    if "prod" in project_id:
        url = "https://mlflow.platform.prod.everycure.org/"
    else:
        url = "https://mlflow.platform.dev.everycure.org/"

    logger.info(f"Auto-detected MLflow URL for project {project_id}: {url}")
    return url


def get_runtime_gcp_bucket(project_id: str = None) -> str:
    """Get the runtime GCP bucket, with smart defaults based on project.

    Args:
        project_id: GCP project ID to use for determining environment

    Returns:
        str: The GCP bucket name to use
    """
    # Check if explicitly set in environment
    env_bucket = environ.get("RUNTIME_GCP_BUCKET")
    if env_bucket:
        logger.debug(f"Using GCP bucket from environment: {env_bucket}")
        return env_bucket

    # Auto-detect based on project ID
    if project_id is None:
        project_id = get_runtime_gcp_project_id()

    # Determine environment from project ID and construct bucket name
    if "prod" in project_id:
        bucket = "mtrx-us-central1-hub-prod-storage"
    else:
        bucket = "mtrx-us-central1-hub-dev-storage"

    logger.info(f"Auto-detected GCP bucket for project {project_id}: {bucket}")
    return bucket
