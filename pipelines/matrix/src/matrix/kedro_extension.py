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

    def model_config(self) -> dict:
        """Pydantic model config."""
        return {"validate_assignment": True}

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
        if self.memory_limit_gb < self.memory_request_gb:
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


class KubernetesNode(Node):
    def __init__(self, *args, k8s_config: KubernetesExecutionConfig = KubernetesExecutionConfig(), **kwargs):
        self.k8s_config = k8s_config
        super().__init__(*args, **kwargs)
