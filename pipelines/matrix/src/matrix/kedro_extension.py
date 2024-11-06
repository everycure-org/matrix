from kedro.pipeline.node import Node
from pydantic import BaseModel


class KubernetesExecutionConfig(BaseModel):
    """Configuration for Kubernetes execution."""

    use_gpu: bool = False


class KubernetesNode(Node):
    def __init__(self, *args, k8s_config: KubernetesExecutionConfig = KubernetesExecutionConfig(), **kwargs):
        self.k8s_config = k8s_config
        super().__init__(*args, **kwargs)
