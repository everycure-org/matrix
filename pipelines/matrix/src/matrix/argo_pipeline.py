from typing import List

from matrix.kedro_extension import ArgoNode


class ArgoPipeline:
    """Argo pipeline."""

    def __init__(self, nodes: List[ArgoNode]):
        self.nodes = nodes

    def get_argo_tasks(self) -> List[ArgoNode]:
        """Get Argo tasks of the pipeline."""
        pass

    def fuse_argo_tasks(self) -> "ArgoPipeline":
        """Fuse two pipelines."""
        pass

    def get_argo_template(self) -> str:
        """Get Argo template of the pipeline."""
        pass
