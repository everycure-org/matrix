from kedro.pipeline.node import Node


class KubernetesNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
