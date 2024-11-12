from typing import Callable, Iterable, Union, Any
import warnings
from kedro.pipeline.node import Node
from pydantic import BaseModel, field_validator, model_validator

# Values are in Gb
KUBERNETES_DEFAULT_LIMIT_RAM = 64
KUBERNETES_DEFAULT_REQUEST_RAM = 64

# Values are in number of cores
KUBERNETES_DEFAULT_LIMIT_CPU = 16
KUBERNETES_DEFAULT_REQUEST_CPU = 4


class ArgoNodeConfig(BaseModel):
    """Configuration for Kubernetes execution.

    Default values are set in settings.py.

    Attributes:
        num_gpus (int): Number of GPUs requested for the container.
        cpu_request (float): CPU cores requested for the container. Written in number of cores.
        cpu_limit (float): Maximum CPU cores allowed for the container. Written in number of cores.
        memory_request (float): Memory requested for the container in GB.
        memory_limit (float): Maximum memory allowed for the container in GB.
    """

    num_gpus: int = 0
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
    def validate_resource_constraints(self) -> "ArgoNodeConfig":
        """Validate that limits are greater than or equal to requests."""
        if self.cpu_limit < self.cpu_request:
            raise ValueError("CPU limit must be greater than or equal to CPU request")
        if self.memory_limit < self.memory_request:
            raise ValueError("Memory limit must be greater than or equal to memory request")
        return self

    @model_validator(mode="after")
    def validate_values_are_sane(self) -> "ArgoNodeConfig":
        """Validate that CPU and memory limits and requests are not too high."""
        if self.cpu_limit > KUBERNETES_DEFAULT_LIMIT_CPU or self.memory_limit > KUBERNETES_DEFAULT_LIMIT_RAM:
            warnings.warn(
                f"CPU (limit: {self.cpu_limit}, request: {self.cpu_request}) and memory (limit: {self.memory_limit}, request: {self.memory_request}) limits and requests are unrealistically high - are you sure they were set in Gb and not in Mi?"
            )
        return self

    def fuse_config(self, argo_config: Union["ArgoNodeConfig", None]) -> None:
        """Fuse in-place with another K8s config."""
        if argo_config is None:
            return
        self.cpu_limit = max(self.cpu_limit, argo_config.cpu_limit)
        self.cpu_request = max(self.cpu_request, argo_config.cpu_request)
        self.memory_limit = max(self.memory_limit, argo_config.memory_limit)
        self.memory_request = max(self.memory_request, argo_config.memory_request)
        self.num_gpus = max(self.num_gpus, argo_config.num_gpus)


class ArgoNode(Node):
    # TODO: Merge this with former FuseNode
    def __init__(self, *args, argo_config: ArgoNodeConfig = ArgoNodeConfig(), **kwargs):
        self._argo_config = argo_config
        super().__init__(*args, **kwargs)

    @property
    def argo_config(self) -> ArgoNodeConfig:
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


def argo_node(
    func: Callable,
    inputs: str | list[str] | dict[str, str] | None,
    outputs: str | list[str] | dict[str, str] | None,
    argo_config: ArgoNodeConfig = ArgoNodeConfig(),
    *,
    name: str | None = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | list[str] | None = None,
    namespace: str | None = None,
):
    return ArgoNode(
        func,
        inputs,
        outputs,
        argo_config=argo_config,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
    )
