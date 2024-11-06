from typing import Callable, Iterable
import warnings
from kedro.pipeline.node import Node
from pydantic import BaseModel, field_validator, model_validator

from matrix.settings import (
    KUBERNETES_DEFAULT_REQUEST_CPU,
    KUBERNETES_DEFAULT_LIMIT_CPU,
    KUBERNETES_DEFAULT_LIMIT_RAM,
    KUBERNETES_DEFAULT_REQUEST_RAM,
)


class KubernetesExecutionConfig(BaseModel):
    """Configuration for Kubernetes execution.

    Attributes:
        use_gpu (bool): Flag to indicate if GPU should be used.
        cpu_request (float): CPU cores requested for the container. Written in number of cores.
        cpu_limit (float): Maximum CPU cores allowed for the container. Written in number of cores.
        memory_request (float): Memory requested for the container in GB.
        memory_limit (float): Maximum memory allowed for the container in GB.
    """

    use_gpu: bool = False
    cpu_request: float = KUBERNETES_DEFAULT_REQUEST_CPU
    cpu_limit: float = KUBERNETES_DEFAULT_LIMIT_CPU
    memory_request: float = KUBERNETES_DEFAULT_REQUEST_RAM
    memory_limit: float = KUBERNETES_DEFAULT_LIMIT_RAM

    model_config = {"validate_assignment": True, "extra": "forbid"}

    @field_validator("cpu_request", "cpu_limit", "memory_request", "memory_limit")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Validate that resource values are positive."""
        if v <= 0:
            raise ValueError("Resource values must be positive")
        return v

    @model_validator(mode="after")
    def validate_resource_constraints(self) -> "KubernetesExecutionConfig":
        """Validate that limits are greater than or equal to requests."""
        if self.cpu_limit < self.cpu_request:
            raise ValueError("CPU limit must be greater than or equal to CPU request")
        if self.memory_limit < self.memory_request:
            raise ValueError("Memory limit must be greater than or equal to memory request")
        return self

    @model_validator(mode="after")
    def validate_values_are_sane(self) -> "KubernetesExecutionConfig":
        """Validate that CPU and memory limits and requests are not too high."""
        if self.cpu_limit > 64 or self.memory_limit > 512:
            warnings.warn(
                f"CPU (limit: {self.cpu_limit}, request: {self.cpu_request}) and memory (limit: {self.memory_limit}, request: {self.memory_request}) limits and requests are unrealistically high - are you sure they were set in Gb and not in Mi?"
            )
        return self

    def fuse_config(self, k8s_config: "KubernetesExecutionConfig") -> None:
        """Fuse in-place with another K8s config."""
        self.cpu_limit = max(self.cpu_limit, k8s_config.cpu_limit)
        self.cpu_request = max(self.cpu_request, k8s_config.cpu_request)
        self.memory_limit = max(self.memory_limit, k8s_config.memory_limit)
        self.memory_request = max(self.memory_request, k8s_config.memory_request)
        self.use_gpu = self.use_gpu or k8s_config.use_gpu


class KubernetesNode(Node):
    # TODO: Merge this with former FuseNode
    def __init__(self, *args, k8s_config: KubernetesExecutionConfig = KubernetesExecutionConfig(), **kwargs):
        self.k8s_config = k8s_config
        super().__init__(*args, **kwargs)

    # TODO: Add fuse() method here.


def kubernetes_node(
    func: Callable,
    inputs: str | list[str] | dict[str, str] | None,
    outputs: str | list[str] | dict[str, str] | None,
    k8s_config: KubernetesExecutionConfig = KubernetesExecutionConfig(),
    *,
    name: str | None = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str | None = None,
):
    return KubernetesNode(
        func,
        inputs,
        outputs,
        k8s_config=k8s_config,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
    )
