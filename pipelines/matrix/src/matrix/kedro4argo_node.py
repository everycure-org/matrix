import os
import warnings
from typing import Any, Optional, Union

from kedro.pipeline.node import Node
from pydantic import BaseModel, field_validator, model_validator

# Values are in GiB
KUBERNETES_DEFAULT_LIMIT_RAM = 52
KUBERNETES_DEFAULT_REQUEST_RAM = 52

# Values are in number of GPUs
KUBERNETES_DEFAULT_NUM_GPUS = 0

# Values are in number of cores
KUBERNETES_DEFAULT_LIMIT_CPU = 14
KUBERNETES_DEFAULT_REQUEST_CPU = 4

SPARK_DRIVER_MEMORY_DEFAULT = int(os.getenv("SPARK_DRIVER_MEMORY", 50))
# FUTURE: Introduce predefined S, M, L, XL resource sizes.


class ArgoResourceConfig(BaseModel):
    """Configuration for Kubernetes execution.

    Default values are set in settings.py.

    Attributes:
        num_gpus (int): Number of GPUs requested for the pod.
        cpu_request (float): CPU cores requested for the pod.
        cpu_limit (float): Maximum CPU cores allowed for the pod.
        memory_request (float): Memory requested for the pod in GB.
        memory_limit (float): Maximum memory allowed for the pod in GB.
    """

    num_gpus: int = KUBERNETES_DEFAULT_NUM_GPUS
    cpu_request: int = KUBERNETES_DEFAULT_REQUEST_CPU
    cpu_limit: int = KUBERNETES_DEFAULT_LIMIT_CPU
    memory_request: int = KUBERNETES_DEFAULT_REQUEST_RAM
    memory_limit: int = KUBERNETES_DEFAULT_LIMIT_RAM
    ephemeral_storage_request: int = 0
    ephemeral_storage_limit: int = 128
    spark_driver_memory: int = SPARK_DRIVER_MEMORY_DEFAULT
    neo4j_heap_size: int = 120

    model_config = {"validate_assignment": True, "extra": "forbid"}

    @field_validator("cpu_request", "cpu_limit", "memory_request", "memory_limit")
    @classmethod
    def validate_integer(cls, v: int) -> int:
        """Validate that resource values are positive integers."""
        if v <= 0:
            raise ValueError("Resource values must be positive")
        if not isinstance(v, int):
            raise ValueError("Currently fractional resource values are not accepted")
        return v

    @model_validator(mode="after")
    def validate_resource_constraints(self) -> "ArgoResourceConfig":
        """Validate that limits are greater than or equal to requests."""
        if self.cpu_limit < self.cpu_request:
            raise ValueError("CPU limit must be greater than or equal to CPU request")
        if self.memory_limit < self.memory_request:
            raise ValueError("Memory limit must be greater than or equal to memory request")
        return self

    @model_validator(mode="after")
    def validate_values_are_sane(self) -> "ArgoResourceConfig":
        """Validate that CPU and memory limits and requests are not too high."""
        if self.cpu_limit > 2 * KUBERNETES_DEFAULT_LIMIT_CPU:
            warnings.warn(
                f"Some of the CPU settings (limit: {self.cpu_limit}, request: {self.cpu_request}) seem quite high - are you sure they are set in cores?"
            )
        if self.memory_limit > 2 * KUBERNETES_DEFAULT_LIMIT_RAM:
            warnings.warn(
                f"Some of the memory settings (limit: {self.memory_limit}, request: {self.memory_request}) are unrealistically high - are you sure they were set in GB and not in Mi?"
            )
        return self

    def fuse_config(self, argo_config: Union["ArgoResourceConfig", None]) -> None:
        """Fuse in-place with another K8s config."""
        if argo_config is None:
            return
        self.cpu_limit = max(self.cpu_limit, argo_config.cpu_limit)
        self.cpu_request = max(self.cpu_request, argo_config.cpu_request)
        self.memory_limit = max(self.memory_limit, argo_config.memory_limit)
        self.memory_request = max(self.memory_request, argo_config.memory_request)
        self.num_gpus = max(self.num_gpus, argo_config.num_gpus)
        self.ephemeral_storage_request = max(self.ephemeral_storage_request, argo_config.ephemeral_storage_request)
        self.ephemeral_storage_limit = max(self.ephemeral_storage_limit, argo_config.ephemeral_storage_limit)
        self.spark_driver_memory = max(self.spark_driver_memory, argo_config.spark_driver_memory)
        self.neo4j_heap_size = self.memory_limit - self.spark_driver_memory

    def model_dump(self, **kwargs) -> dict:
        """Customize JSON or dict export with Kubernetes-compatible formatting."""
        data = super().model_dump(**kwargs)
        data["memory_request"] = f"{int(self.memory_request)}Gi"
        data["memory_limit"] = f"{int(self.memory_limit)}Gi"
        data["ephemeral_storage_request"] = f"{int(self.ephemeral_storage_request)}Gi"
        data["ephemeral_storage_limit"] = f"{int(self.ephemeral_storage_limit)}Gi"
        data["spark_driver_memory"] = f"{int(self.spark_driver_memory)}g"
        data["neo4j_heap_size"] = f"{int(self.spark_driver_memory)}G"
        return data


class ArgoNode(Node):
    # TODO: Merge this with former FuseNode
    def __init__(self, *args, argo_config: Optional[ArgoResourceConfig] = None, **kwargs):
        if argo_config is None:
            argo_config = ArgoResourceConfig()
        self._argo_config = argo_config
        super().__init__(*args, **kwargs)

    @property
    def argo_config(self) -> ArgoResourceConfig:
        return self._argo_config

    # TODO: Add fuse() method here.

    def _copy(self, **overwrite_params: Any) -> "ArgoNode":
        """
        Helper function to copy the node, replacing some values.
        """
        params = {
            "func": self._func,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "argo_config": self._argo_config,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
        }
        params.update(overwrite_params)
        return ArgoNode(**params)  # type: ignore[arg-type]


ARGO_GPU_NODE_MEDIUM = ArgoResourceConfig(num_gpus=1)
