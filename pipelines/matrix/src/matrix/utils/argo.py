import json

from matrix_auth.system import run_subprocess


def argo_lint(file_path: str, verbose: bool) -> str:
    """Lint a Workflow manifest (not a WorkflowTemplate)."""
    run_subprocess(
        f"argo lint {file_path}",
        check=True,
        stream_output=verbose,
    )


def submit_workflow_from_file(file_path: str, namespace: str, verbose: bool) -> str:
    """Submit a workflow from a YAML file directly. This can be done from kubectl"""
    cmd = " ".join(["kubectl create", f"-n {namespace}", f"-f {file_path}", "-o", "json"])

    result = run_subprocess(cmd)
    job_name = json.loads(result.stdout).get("metadata", {}).get("name")

    if not job_name:
        raise RuntimeError("Failed to retrieve job name from Argo submission.")

    return job_name
