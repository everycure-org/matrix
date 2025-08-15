import json

from matrix.utils.system import run_subprocess


def argo_template_lint(file_path: str, verbose: bool) -> str:
    run_subprocess(
        f"argo template lint {file_path}",
        check=True,
        stream_output=verbose,
    )


def submit_workflow(run_name: str, namespace: str, verbose: bool):
    """Submit the Argo workflow and provide instructions for watching."""
    cmd = " ".join(
        [
            "argo submit",
            f"--name {run_name}",
            f"-n {namespace}",
            f"--from wftmpl/{run_name}",  # name of the template resource (created in previous step)
            f"-p run_name={run_name}",
            "-l submit-from-ui=false",
            "-o json",
        ]
    )

    result = run_subprocess(cmd)
    job_name = json.loads(result.stdout).get("metadata", {}).get("name")

    if not job_name:
        raise RuntimeError("Failed to retrieve job name from Argo submission.")

    return job_name
