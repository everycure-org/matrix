import logging
import subprocess
from os import environ
from pathlib import Path

from matrix.utils.system import run_subprocess

logger = logging.getLogger(__name__)


def can_talk_to_kubernetes(
    project: str,
    region: str,
    cluster_name: str,
) -> bool:
    """Check if one can communicate with the Kubernetes cluster, using the kubectl CLI.

    If kubectl is not installed, it attempts to install and configure it using gcloud components.

    Raises:
        EnvironmentError: If gcloud is not installed or kubectl cannot be configured.
    """

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
