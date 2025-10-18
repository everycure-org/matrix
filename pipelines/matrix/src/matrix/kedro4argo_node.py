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
        ephemeral_storage_request (float): Ephemeral storage requested for the pod in GB.
        ephemeral_storage_limit (float): Maximum ephemeral storage allowed for the pod in GB.

        The above ephemeral storage parameters are related to, and work in conjunction with
        ephemeral-storage-local-ssd parameter on the Node Pools (1st or 2nd gen) - refer to gke.tf for how it's applied.
        More information here: https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/local-ssd#1st-or-2nd-generation_1
        and here: https://cloud.google.com/compute/docs/general-purpose-machines#n2d-high-mem
    """

    num_gpus: int = KUBERNETES_DEFAULT_NUM_GPUS
    cpu_request: int = KUBERNETES_DEFAULT_REQUEST_CPU
    cpu_limit: int = KUBERNETES_DEFAULT_LIMIT_CPU
    memory_request: int = KUBERNETES_DEFAULT_REQUEST_RAM
    memory_limit: int = KUBERNETES_DEFAULT_LIMIT_RAM
    ephemeral_storage_request: int = 0
    ephemeral_storage_limit: int = 128

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
        if self.cpu_limit > 10 * KUBERNETES_DEFAULT_LIMIT_CPU:
            warnings.warn(
                f"Some of the CPU settings (limit: {self.cpu_limit}, request: {self.cpu_request}) seem quite high - are you sure they are set in cores?"
            )
        if self.memory_limit > 10 * KUBERNETES_DEFAULT_LIMIT_RAM:
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

    def model_dump(self, **kwargs) -> dict:
        """Customize JSON or dict export with Kubernetes-compatible formatting."""
        data = super().model_dump(**kwargs)
        data["memory_request"] = int(self.memory_request)
        data["memory_limit"] = int(self.memory_limit)
        data["ephemeral_storage_request"] = int(self.ephemeral_storage_request)
        data["ephemeral_storage_limit"] = int(self.ephemeral_storage_limit)
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
ARGO_NODE_MEDIUM_MATRIX_GENERATION = ArgoResourceConfig(
    cpu_limit=48,
    cpu_request=48,
    ephemeral_storage_limit=512,
    ephemeral_storage_request=512,
    memory_limit=256,
    memory_request=256,
)

ARGO_NODE_LARGE_MEMORY_MATRIX_GENERATION = ArgoResourceConfig(
    cpu_limit=65,
    cpu_request=65,
    ephemeral_storage_limit=512,
    ephemeral_storage_request=512,
    memory_limit=256,
    memory_request=256,
)

ARGO_CPU_ONLY_NODE_MEDIUM = ArgoResourceConfig()
