"""
# NOTE: This file was partially generated using AI assistance.

Hera-based Argo Workflow generator for Kedro pipelines.

This module provides a Pythonic way to generate Argo Workflows using the Hera library,
replacing the previous Jinja templating approach while maintaining compatibility with
the existing Kedro fusion logic and resource configuration system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Hera imports (local vendored hera present under /Users/pascal/Code/others/hera)
from hera.workflows import (  # type: ignore[import]
    DAG,
    ConfigMapEnv,
    Container,
    Env,
    Label,
    Metrics,
    Parameter,
    Resources,
    SecretEnv,
    Workflow,
)
from hera.workflows.metrics import Counter  # type: ignore[import]
from hera.workflows.task import Task  # type: ignore[import]
from hera.workflows.user_container import UserContainer  # type: ignore[import]
from hera.workflows.volume import SecretVolume  # type: ignore[import]
from kedro.pipeline import Pipeline  # type: ignore[import]

from matrix.argo import fuse, get_dependencies, get_trigger_release_flag
from matrix.git_utils import get_git_sha
from matrix.kedro4argo_node import ArgoResourceConfig

# Constants
DEFAULT_NEO4J_IMAGE = "neo4j:5.21.0-enterprise"
DEFAULT_NEO4J_HEAP_RATIO = 0.7
DEFAULT_NEO4J_MEMORY_GB = 60
DEFAULT_SPARK_DRIVER_MEMORY = "60"
NEO4J_LOCAL_HOST = "bolt://127.0.0.1:7687"
NEO4J_CLUSTER_HOST = "bolt+ssc://neo4j.neo4j.svc.cluster.local:7687"
MLFLOW_ENDPOINT = "http://mlflow-tracking.mlflow.svc.cluster.local:80"
OPENAI_ENDPOINT = "https://api.openai.com/v1"


@dataclass
class WorkflowConfig:
    """Configuration for workflow generation."""

    image: str
    run_name: str
    release_version: str
    mlflow_experiment_id: int
    mlflow_url: str
    namespace: str
    username: str
    pipeline: Pipeline
    environment: str
    package_name: str
    release_folder_name: str
    default_execution_resources: ArgoResourceConfig
    mlflow_run_id: Optional[str] = None


class EnvironmentBuilder:
    """Builder for workflow environment variables."""

    @staticmethod
    def build_base_env(config: WorkflowConfig) -> List[Env | SecretEnv | ConfigMapEnv]:
        """Build the base environment variables for all containers.

        Args:
            config: Workflow configuration

        Returns:
            List of environment variable configurations
        """
        include_private = os.getenv("INCLUDE_PRIVATE_DATASETS", "0")

        return [
            # Standard environment variables
            Env(name="WORKFLOW_ID", value="{{workflow.name}}"),
            Env(name="RUN_NAME", value=config.run_name),
            Env(name="RELEASE_VERSION", value=config.release_version),
            Env(name="RELEASE_FOLDER_NAME", value=config.release_folder_name),
            Env(name="NEO4J_HOST", value=NEO4J_CLUSTER_HOST),
            Env(name="MLFLOW_ENDPOINT", value=MLFLOW_ENDPOINT),
            Env(name="MLFLOW_EXPERIMENT_ID", value=str(config.mlflow_experiment_id)),
            Env(name="MLFLOW_RUN_ID", value=config.mlflow_run_id or ""),
            Env(name="MLFLOW_URL", value=config.mlflow_url),
            Env(name="INCLUDE_PRIVATE_DATASETS", value=include_private),
            Env(name="OPENAI_ENDPOINT", value=OPENAI_ENDPOINT),
            # Secrets
            SecretEnv(name="OPENAI_API_KEY", secret_name="matrix-secrets", secret_key="OPENAI_API_KEY"),
            SecretEnv(name="NEO4J_USER", secret_name="matrix-secrets", secret_key="NEO4J_USER"),
            SecretEnv(name="NEO4J_PASSWORD", secret_name="matrix-secrets", secret_key="NEO4J_PASSWORD"),
            SecretEnv(name="GH_TOKEN", secret_name="gh-password", secret_key="GH_CREDS"),
            # ConfigMap values
            ConfigMapEnv(
                name="RUNTIME_GCP_PROJECT_ID", config_map_name="matrix-config", config_map_key="GCP_PROJECT_ID"
            ),
            ConfigMapEnv(name="RUNTIME_GCP_BUCKET", config_map_name="matrix-config", config_map_key="GCP_BUCKET"),
            # Resource-based environment variables
            Env(name="NUM_CPU", value="{{inputs.parameters.cpu_limit}}"),
            Env(name="SPARK_DRIVER_MEMORY", value="{{inputs.parameters.memory_limit}}"),
        ]

    @staticmethod
    def build_neo4j_env(config: WorkflowConfig) -> List[Env | SecretEnv | ConfigMapEnv]:
        """Build environment variables for Neo4j containers.

        Args:
            config: Workflow configuration

        Returns:
            List of environment variable configurations for Neo4j
        """
        base_env = EnvironmentBuilder.build_base_env(config)
        return base_env + [
            Env(name="SPARK_DRIVER_MEMORY", value=DEFAULT_SPARK_DRIVER_MEMORY),
            Env(name="NEO4J_HOST", value=NEO4J_LOCAL_HOST),
        ]


class ResourceConverter:
    """Converter for resource configurations."""

    @staticmethod
    def to_hera_resources(resource_cfg: Dict[str, Any]) -> Resources:
        """Convert resource configuration to Hera Resources.

        Args:
            resource_cfg: Dictionary with resource configuration

        Returns:
            Hera Resources object
        """
        return Resources(
            cpu_request=str(resource_cfg["cpu_request"]),
            cpu_limit=str(resource_cfg["cpu_limit"]),
            memory_request=f"{int(resource_cfg['memory_request'])}Gi",
            memory_limit=f"{int(resource_cfg['memory_limit'])}Gi",
            ephemeral_request=f"{int(resource_cfg['ephemeral_storage_request'])}Gi",
            ephemeral_limit=f"{int(resource_cfg['ephemeral_storage_limit'])}Gi",
            gpus=str(resource_cfg["num_gpus"]),
        )

    @staticmethod
    def format_memory(value: Any) -> str:
        """Format memory value with Gi suffix.

        Args:
            value: Memory value (int or str)

        Returns:
            Formatted memory string with Gi suffix
        """
        return f"{int(value)}Gi"

    @staticmethod
    def calculate_neo4j_heap(memory_limit: Any) -> int:
        """Calculate Neo4j heap size based on memory limit.

        Args:
            memory_limit: Memory limit value

        Returns:
            Heap size in GB
        """
        try:
            mem_lim_gi = int(str(memory_limit))
        except (ValueError, TypeError):
            mem_str = str(memory_limit)
            if mem_str.endswith("Gi"):
                mem_lim_gi = int(mem_str.rstrip("Gi"))
            else:
                mem_lim_gi = DEFAULT_NEO4J_MEMORY_GB

        return max(int(mem_lim_gi * DEFAULT_NEO4J_HEAP_RATIO), 1)


class TemplateBuilder:
    """Builder for workflow templates."""

    def __init__(self, config: WorkflowConfig):
        """Initialize template builder.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self.base_env = EnvironmentBuilder.build_base_env(config)
        self.neo4j_env = EnvironmentBuilder.build_neo4j_env(config)

    def build_default_container(self) -> Container:
        """Build the default Kedro container template.

        Returns:
            Container template for standard Kedro tasks
        """
        return Container(
            name="kedro-task",
            image=self.config.image,
            image_pull_policy="Always",
            env=self.base_env,
            command=["kedro"],
            args=[
                "run",
                "-p",
                "{{inputs.parameters.pipeline_name}}",
                "-e",
                "{{inputs.parameters.environment}}",
                "-n",
                "{{inputs.parameters.kedro_nodes}}",
            ],
            pod_spec_patch=self._get_pod_spec_patch(),
            labels=self._get_container_labels(),
            tolerations=self._get_tolerations(),
            inputs=self._get_default_inputs(),
        )

    def build_neo4j_container(self) -> Container:
        """Build the Neo4j container template with sidecar.

        Returns:
            Container template for Neo4j tasks
        """
        secret_vol = SecretVolume(
            name="gds-key",
            secret_name="gds-secret",
            items=[{"key": "gds_key", "path": "gds-key"}],
            mount_path="/licences",
            read_only=True,
        )

        neo4j_sidecar = self._build_neo4j_sidecar()

        return Container(
            name="kedro-task-neo4j",
            image=self.config.image,
            image_pull_policy="Always",
            env=self.neo4j_env,
            command=["/bin/sh", "-c"],
            args=[self._get_neo4j_startup_script()],
            pod_spec_patch=self._get_pod_spec_patch(),
            labels=self._get_container_labels(template="neo4j"),
            tolerations=self._get_tolerations(),
            inputs=self._get_neo4j_inputs(),
            volumes=[secret_vol],
            sidecars=[neo4j_sidecar],
        )

    def _build_neo4j_sidecar(self) -> UserContainer:
        """Build Neo4j sidecar container.

        Returns:
            Neo4j sidecar container configuration
        """
        secret_vol = SecretVolume(
            name="gds-key",
            secret_name="gds-secret",
            items=[{"key": "gds_key", "path": "gds-key"}],
            mount_path="/licences",
            read_only=True,
        )

        return UserContainer(
            name="neo4j",
            image=DEFAULT_NEO4J_IMAGE,
            env=[
                SecretEnv(name="NEO4J_AUTH", secret_name="matrix-secrets", secret_key="NEO4J_AUTH"),
                Env(name="NEO4J_gds_enterprise_license__file", value="/licences/gds-key"),
                Env(name="NEO4J_apoc_export_file_enabled", value="true"),
                Env(name="NEO4J_apoc_import_file_enabled", value="true"),
                Env(name="NEO4J_apoc_import_file_use__neo4j__config", value="true"),
                Env(name="NEO4J_PLUGINS", value='["apoc", "graph-data-science", "apoc-extended"]'),
                Env(name="NEO4J_dbms_security_auth__minimum__password__length", value="4"),
                Env(name="NEO4J_dbms_security_procedures_whitelist", value="gds.*, apoc.*"),
                Env(name="NEO4J_dbms_security_procedures_unrestricted", value="gds.*, apoc.*"),
                Env(name="NEO4J_db_logs_query_enabled", value="OFF"),
                Env(name="NEO4J_ACCEPT_LICENSE_AGREEMENT", value="yes"),
                Env(name="NEO4J_dbms_memory_heap_initial__size", value="{{inputs.parameters.neo4j_heap_g}}G"),
                Env(name="NEO4J_dbms_memory_heap_max__size", value="{{inputs.parameters.neo4j_heap_g}}G"),
            ],
            volumes=[secret_vol],
        )

    def _get_neo4j_startup_script(self) -> str:
        """Get Neo4j startup script.

        Returns:
            Shell script for Neo4j startup
        """
        return (
            "echo 'Waiting for Neo4j to be ready...' && "
            "until curl -s http://localhost:7474/ready; do echo 'Waiting...'; sleep 5; done; "
            "echo 'Neo4j is ready. Starting main application...' && "
            "kedro run -p {{inputs.parameters.pipeline_name}} "
            "-e {{inputs.parameters.environment}} -n {{inputs.parameters.kedro_nodes}}"
        )

    def _get_pod_spec_patch(self) -> str:
        """Get pod spec patch for resource configuration.

        Returns:
            YAML string for pod spec patch
        """
        return """\
tolerations:
  - key: "node-memory-size"
    operator: "Equal"
    value: "large"
    effect: "NoSchedule"
  - key: "workload"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
containers:
  - name: main
    resources:
      requests:
        memory: "{{inputs.parameters.memory_request}}Gi"
        cpu: "{{inputs.parameters.cpu_request}}"
        nvidia.com/gpu: "{{inputs.parameters.num_gpus}}"
        ephemeral-storage: "{{inputs.parameters.ephemeral_storage_request}}Gi"
      limits:
        memory: "{{inputs.parameters.memory_limit}}Gi"
        cpu: "{{inputs.parameters.cpu_limit}}"
        nvidia.com/gpu: "{{inputs.parameters.num_gpus}}"
        ephemeral-storage: "{{inputs.parameters.ephemeral_storage_limit}}Gi"
"""

    def _get_container_labels(self, template: Optional[str] = None) -> Dict[str, str]:
        """Get container labels.

        Args:
            template: Optional template name for additional labeling

        Returns:
            Dictionary of labels
        """
        labels = {
            "app": "kedro-argo",
            "workflow_name": self.config.run_name,
            "kedro_nodes": "{{inputs.parameters.kedro_nodes_trimmed}}",
            "cost-center": "matrix-pipeline",
            "environment": "{{inputs.parameters.environment}}",
        }
        if template:
            labels["template"] = template
        return labels

    def _get_tolerations(self) -> List[Dict[str, str]]:
        """Get pod tolerations.

        Returns:
            List of toleration configurations
        """
        return [
            {"key": "workload", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
            {"key": "node-memory-size", "operator": "Equal", "value": "large", "effect": "NoSchedule"},
        ]

    def _get_default_inputs(self) -> List[Parameter]:
        """Get default input parameters.

        Returns:
            List of parameter definitions
        """
        return [
            Parameter(name="kedro_nodes"),
            Parameter(name="pipeline_name"),
            Parameter(name="environment"),
            Parameter(name="kedro_nodes_trimmed"),
            Parameter(name="cpu_request"),
            Parameter(name="cpu_limit"),
            Parameter(name="memory_request"),
            Parameter(name="memory_limit"),
            Parameter(name="ephemeral_storage_request"),
            Parameter(name="ephemeral_storage_limit"),
            Parameter(name="num_gpus"),
        ]

    def _get_neo4j_inputs(self) -> List[Parameter]:
        """Get Neo4j input parameters.

        Returns:
            List of parameter definitions including Neo4j-specific ones
        """
        return self._get_default_inputs() + [Parameter(name="neo4j_heap_g")]


class TaskBuilder:
    """Builder for workflow tasks."""

    def __init__(self, config: WorkflowConfig, templates: Dict[str, Container]):
        """Initialize task builder.

        Args:
            config: Workflow configuration
            templates: Dictionary of container templates
        """
        self.config = config
        self.templates = templates

    def build_task(self, task_config: Dict[str, Any]) -> Task:
        """Build a task from configuration.

        Args:
            task_config: Task configuration from pipeline

        Returns:
            Configured Task object
        """
        kedro_nodes = task_config["nodes"]
        trimmed = self._trim_node_name(kedro_nodes)
        resources = task_config["resources"]

        arguments = self._build_arguments(kedro_nodes, trimmed, resources)

        # Select template and add Neo4j-specific configuration if needed
        if task_config.get("template") == "neo4j":
            heap_g = ResourceConverter.calculate_neo4j_heap(resources["memory_limit"])
            arguments["neo4j_heap_g"] = str(heap_g)
            template = self.templates["neo4j"]
        else:
            template = self.templates["default"]

        return Task(name=task_config["name"], template=template, arguments=arguments)

    def _build_arguments(self, kedro_nodes: str, trimmed: str, resources: Dict[str, Any]) -> Dict[str, str]:
        """Build task arguments.

        Args:
            kedro_nodes: Kedro node names
            trimmed: Trimmed node name for display
            resources: Resource configuration

        Returns:
            Dictionary of arguments
        """
        return {
            "kedro_nodes": kedro_nodes,
            "kedro_nodes_trimmed": trimmed,
            "pipeline_name": self.config.pipeline.name,
            "environment": self.config.environment,
            "cpu_request": str(resources["cpu_request"]),
            "cpu_limit": str(resources["cpu_limit"]),
            "memory_request": str(resources["memory_request"]),
            "memory_limit": str(resources["memory_limit"]),
            "ephemeral_storage_request": str(resources["ephemeral_storage_request"]),
            "ephemeral_storage_limit": str(resources["ephemeral_storage_limit"]),
            "num_gpus": str(resources["num_gpus"]),
        }

    def _trim_node_name(self, node_name: str, max_length: int = 63) -> str:
        """Trim node name for Kubernetes compatibility.

        Args:
            node_name: Original node name
            max_length: Maximum allowed length

        Returns:
            Trimmed and sanitized node name
        """
        return node_name[:max_length].replace(",", "-").rstrip("_-.")


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

    This function generates an Argo Workflow using the Hera library,
    maintaining compatibility with the legacy template-based approach.

    Args:
        image: Docker image to use
        run_name: Name of the workflow run
        release_version: Version of the release
        mlflow_experiment_id: MLflow experiment ID
        mlflow_url: MLflow tracking URL
        namespace: Kubernetes namespace
        username: Username for tracking
        pipeline: Kedro pipeline to execute
        environment: Kedro environment name
        package_name: Package name (for metadata)
        release_folder_name: Folder name for releases
        default_execution_resources: Default resource configuration
        mlflow_run_id: Optional MLflow run ID

    Returns:
        YAML string of the workflow
    """
    if default_execution_resources is None:
        default_execution_resources = ArgoResourceConfig()

    # Create configuration object
    config = WorkflowConfig(
        image=image,
        run_name=run_name,
        release_version=release_version,
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_url=mlflow_url,
        namespace=namespace,
        username=username,
        pipeline=pipeline,
        environment=environment,
        package_name=package_name,
        release_folder_name=release_folder_name,
        default_execution_resources=default_execution_resources,
        mlflow_run_id=mlflow_run_id,
    )

    # Process pipeline
    fused = fuse(pipeline)
    pipeline_tasks = get_dependencies(fused, default_execution_resources)

    # Build workflow
    wf = _build_workflow(config, pipeline_tasks)

    # FUTURE: This is where we should in the future move towrads
    # submitting the workflow directly from python instead of rendering the yaml
    # and submitting via the argo binary

    return wf.to_yaml()


def _build_workflow(config: WorkflowConfig, pipeline_tasks: List[Dict[str, Any]]) -> Workflow:
    """Build the workflow object.

    Args:
        config: Workflow configuration
        pipeline_tasks: List of pipeline task configurations

    Returns:
        Configured Workflow object
    """
    # Create workflow with metadata
    wf = Workflow(
        name=config.run_name,
        namespace=config.namespace,
        labels=_build_workflow_labels(config),
        metrics=_build_workflow_metrics(config.pipeline.name),
    )

    with wf:
        # Set workflow arguments
        wf.arguments = _build_workflow_arguments(config)

        # Build templates
        template_builder = TemplateBuilder(config)
        templates = {
            "default": template_builder.build_default_container(),
            "neo4j": template_builder.build_neo4j_container(),
        }

        # Build DAG with tasks
        _build_dag(config, pipeline_tasks, templates)

        wf.entrypoint = "main"

    return wf


def _build_workflow_labels(config: WorkflowConfig) -> Dict[str, str]:
    """Build workflow labels.

    Args:
        config: Workflow configuration

    Returns:
        Dictionary of labels
    """
    return {
        "run": config.run_name,
        "workflow_name": config.run_name,
        "username": config.username,
        "trigger_release": get_trigger_release_flag(config.pipeline.name),
        "release_version": config.release_version,
        "git_sha": get_git_sha(),
    }


def _build_workflow_metrics(pipeline_name: str) -> Metrics:
    """Build workflow metrics configuration.

    Args:
        pipeline_name: Name of the pipeline

    Returns:
        Metrics configuration
    """
    return Metrics(
        metrics=[
            Counter(
                name="argo_custom_workflow_error_counter",
                help="Total number of failed workflows",
                labels=[Label(key="pipeline_name", value=pipeline_name)],
                when="{{ status }} == Failed",
                value="1",
            )
        ]
    )


def _build_workflow_arguments(config: WorkflowConfig) -> Dict[str, List[Parameter]]:
    """Build workflow arguments.

    Args:
        config: Workflow configuration

    Returns:
        Dictionary with parameter list
    """
    include_private = os.getenv("INCLUDE_PRIVATE_DATASETS", "0")

    return {
        "parameters": [
            Parameter(name="image", value=config.image),
            Parameter(name="run_name", value=config.run_name),
            Parameter(name="pipeline_name", value=config.pipeline.name),
            Parameter(name="env", value=config.environment),
            Parameter(name="release_folder_name", value=config.release_folder_name),
            Parameter(name="mlflow_experiment_id", value=str(config.mlflow_experiment_id)),
            Parameter(name="mlflow_run_id", value=config.mlflow_run_id or ""),
            Parameter(name="mlflow_url", value=config.mlflow_url),
            Parameter(name="include_private_datasets", value=include_private),
        ]
    }


def _build_dag(config: WorkflowConfig, pipeline_tasks: List[Dict[str, Any]], templates: Dict[str, Container]) -> None:
    """Build the DAG with tasks and dependencies.

    Args:
        config: Workflow configuration
        pipeline_tasks: List of pipeline task configurations
        templates: Dictionary of container templates
    """
    task_builder = TaskBuilder(config, templates)
    name_to_task: Dict[str, Task] = {}

    with DAG(name="main"):
        # Create tasks
        for task_config in pipeline_tasks:
            task = task_builder.build_task(task_config)
            name_to_task[task_config["name"]] = task

        # Set up dependencies
        for task_config in pipeline_tasks:
            for dep in task_config["deps"]:
                name_to_task[dep] >> name_to_task[task_config["name"]]
