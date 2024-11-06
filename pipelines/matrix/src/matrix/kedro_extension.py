from kedro.pipeline.node import Node
from pydantic import BaseModel


class KubernetesExecutionConfig(BaseModel):
    """Configuration for Kubernetes execution."""

    use_gpu: bool = False


class KubernetesNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
