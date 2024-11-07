from typing import List

from matrix.kedro_extension import ArgoNode
from kedro.pipeline import Pipeline


class ArgoPipeline:
    """Argo pipeline.

    A class that represents an Argo workflow pipeline composed of ArgoNodes.

    Args:
        nodes: List of ArgoNode objects representing the pipeline tasks.
    """

    def __init__(self, pipeline: Pipeline):
        self.nodes = pipeline.nodes

    def __len__(self) -> int:
        """Get the number of nodes in the pipeline."""
        return len(self.nodes)

    @property
    def tasks(self) -> List[ArgoNode]:
        """Get Argo tasks of the pipeline.

        Returns:
            List[ArgoNode]: List of ArgoNode objects representing pipeline tasks.
        """
        return self.nodes

    def kedro_command(self) -> str:
        """Get the Kedro command for executing the pipeline.

        Returns:
            str: Kedro command string for pipeline execution.
        """
        return "kedro run"  # Basic implementation - may need to be enhanced based on requirements

    def fuse_argo_tasks(self) -> "ArgoPipeline":
        """Fuse two pipelines."""
        pass

    def get_argo_template(self) -> str:
        """Get Argo template of the pipeline."""
        pass
