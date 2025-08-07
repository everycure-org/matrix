"""
# NOTE: This file was partially generated using AI assistance.

Hera-based Argo Workflow generator and submitter.

This module replaces the Jinja templating approach with a Pythonic
composition of an Argo Workflow using `hera.workflows`. It preserves
the Kedro fusion logic and resource overrides via `ArgoResourceConfig`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Hera imports (local vendored hera present under /Users/pascal/Code/others/hera)
from hera.workflows import DAG, Container, Env, Metric, Metrics, Parameter, Resources, SecretEnv, Workflow
from kedro.pipeline import Pipeline

from matrix.argo import fuse, get_dependencies, get_trigger_release_flag
from matrix.git_utils import get_git_sha
from matrix.kedro4argo_node import ArgoResourceConfig


def _build_base_env(
    run_name: str,
    release_version: str,
    release_folder_name: str,
    mlflow_experiment_id: int,
    mlflow_run_id: Optional[str],
    mlflow_url: str,
) -> List[Env | SecretEnv]:
    mlflow_run_id_value = mlflow_run_id or ""
    include_private = os.getenv("INCLUDE_PRIVATE_DATASETS", "0")

    # ConfigMap env cannot be expressed directly as Env in Hera v5, keep to explicit envs or wiring in pod spec patch if needed
    base_env: List[Env | SecretEnv] = [
        Env(name="RUN_NAME", value=run_name),
        Env(name="RELEASE_VERSION", value=release_version),
        Env(name="RELEASE_FOLDER_NAME", value=release_folder_name),
        Env(name="NEO4J_HOST", value="bolt+ssc://neo4j.neo4j.svc.cluster.local:7687"),
        Env(name="MLFLOW_ENDPOINT", value="http://mlflow-tracking.mlflow.svc.cluster.local:80"),
        Env(name="MLFLOW_EXPERIMENT_ID", value=str(mlflow_experiment_id)),
        Env(name="MLFLOW_RUN_ID", value=mlflow_run_id_value),
        Env(name="MLFLOW_URL", value=mlflow_url),
        Env(name="INCLUDE_PRIVATE_DATASETS", value=include_private),
        Env(name="OPENAI_ENDPOINT", value="https://api.openai.com/v1"),
        SecretEnv(name="OPENAI_API_KEY", secret_name="matrix-secrets", secret_key="OPENAI_API_KEY"),
        SecretEnv(name="NEO4J_USER", secret_name="matrix-secrets", secret_key="NEO4J_USER"),
        SecretEnv(name="NEO4J_PASSWORD", secret_name="matrix-secrets", secret_key="NEO4J_PASSWORD"),
        # RUNTIME_GCP_* come from ConfigMap via envFrom in YAML; Hera lacks a direct ConfigMapKeyRef helper here.
        # Fall back to reading them via the running environment since Argo injects the ConfigMap via envFrom at operator side.
        # If not present, tasks should still work provided downstream code resolves project/bucket via config.
    ]
    return base_env


def _to_hera_resources(resource_cfg: Dict[str, Any]) -> Resources:
    return Resources(
        cpu_request=str(resource_cfg["cpu_request"]),
        cpu_limit=str(resource_cfg["cpu_limit"]),
        memory_request=f"{int(resource_cfg['memory_request'])}Gi",
        memory_limit=f"{int(resource_cfg['memory_limit'])}Gi",
        ephemeral_request=f"{int(resource_cfg['ephemeral_storage_request'])}Gi",
        ephemeral_limit=f"{int(resource_cfg['ephemeral_storage_limit'])}Gi",
        gpus=str(resource_cfg["num_gpus"]),
    )


def build_workflow_yaml(
    *,
    image: str,
    run_name: str,
    release_version: str,
    mlflow_experiment_id: int,
    mlflow_url: str,
    namespace: str,
    username: str,
    pipeline: Pipeline,
    environment: str,
    package_name: str,
    release_folder_name: str,
    default_execution_resources: Optional[ArgoResourceConfig] = None,
    mlflow_run_id: Optional[str] = None,
) -> str:
    """Construct a Hera Workflow YAML for direct submission.

    Args mirror the legacy `generate_argo_config` for parity.
    """
    if default_execution_resources is None:
        default_execution_resources = ArgoResourceConfig()

    fused = fuse(pipeline)
    pipeline_tasks = get_dependencies(fused, default_execution_resources)

    # Global labels/metadata
    labels = {
        "run": run_name,
        "workflow_name": run_name,
        "username": username,
        "pipeline_name": pipeline.name,
        "trigger_release": get_trigger_release_flag(pipeline.name),
        "release_version": release_version,
        "git_sha": get_git_sha(),
    }

    # Prometheus metrics parity
    metrics = Metrics(
        prometheus=[
            Metric(
                name="argo_custom_workflow_error_counter",
                help="Total number of failed workflows",
                labels={"pipeline_name": pipeline.name},
                when="{{ status }}  == Failed",
                value="1",
            )
        ]
    )

    wf = Workflow(name=run_name, namespace=namespace, labels=labels, metrics=metrics)

    # Input parameters parity (useful for UI visibility)
    wf.parameters = [
        Parameter(name="image", value=image),
        Parameter(name="run_name", value=run_name),
        Parameter(name="pipeline_name", value=pipeline.name),
        Parameter(name="env", value=environment),
        Parameter(name="release_folder_name", value=release_folder_name),
        Parameter(name="mlflow_experiment_id", value=str(mlflow_experiment_id)),
        Parameter(name="mlflow_run_id", value=mlflow_run_id or ""),
        Parameter(name="mlflow_url", value=mlflow_url),
        Parameter(name="include_private_datasets", value=os.getenv("INCLUDE_PRIVATE_DATASETS", "0")),
    ]

    base_env = _build_base_env(
        run_name=run_name,
        release_version=release_version,
        release_folder_name=release_folder_name,
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_run_id=mlflow_run_id,
        mlflow_url=mlflow_url,
    )

    name_to_task: Dict[str, Any] = {}
    tasks_in_order: List[Any] = []

    for task in pipeline_tasks:
        kedro_nodes = task["nodes"]
        trimmed = kedro_nodes[:63].replace(",", "-").rstrip("_-.")
        task_labels = {
            "app": "kedro-argo",
            "workflow_name": run_name,
            "kedro_nodes": trimmed,
            "cost-center": "matrix-pipeline",
            "environment": environment,
            "pipeline": pipeline.name,
        }
        resources = _to_hera_resources(task["resources"])

        container = Container(
            name="main",
            image=image,
            image_pull_policy="Always",
            env=base_env,
            command=["kedro"],
            args=["run", "-p", pipeline.name, "-e", environment, "-n", kedro_nodes],
            resources=resources,
            labels=task_labels,
            tolerations=[
                {"key": "workload", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
                {"key": "node-memory-size", "operator": "Equal", "value": "large", "effect": "NoSchedule"},
            ],
        )

        t = container.as_task(name=task["name"])  # type: ignore[attr-defined]
        name_to_task[task["name"]] = t
        tasks_in_order.append(t)

    # Wire DAG deps
    dag = DAG(tasks=tasks_in_order)
    for task in pipeline_tasks:
        for dep in task["deps"]:
            name_to_task[dep] >> name_to_task[task["name"]]

    wf.add(dag)
    return wf.to_yaml()
