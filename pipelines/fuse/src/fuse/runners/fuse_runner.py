from typing import Iterable, List

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner.sequential_runner import SequentialRunner
from pluggy import PluginManager


class FusedNode(Node):
    """FusedNode is an extension of Kedro's internal node. The FusedNode
    wraps a set of nodes, and correctly sets it's `inputs` and `outputs`
    allowing it to act as a single unit for execution.
    """

    def __init__(self, nodes: List[Node], name: str):
        self._nodes = nodes
        self._name = name
        self._namespace = None
        self._inputs = []
        self._outputs = []
        self._confirms = []
        self._func = lambda: None
        self._tags = []

        for node in nodes:
            self._inputs.extend(node.inputs)
            self._outputs.extend(node.outputs)
            self._tags.extend(node._tags)

        # NOTE: Exclude ouputs made as part of the intermediate nodes
        for node in self._outputs:
            if node in self._inputs:
                self._inputs.remove(node)

        self._tags = list(set(self._tags))


class FusedPipeline(Pipeline):
    """Fused pipeline allows for wrapping nodes for execution by the underlying
    pipeline execution framework.

    This is needed, as Kedro immediately translates a pipeline to a list of nodes
    to execute, where any pipeline structure is flatmapped. The FusedPipeline produces
    a _single_ FusedNode that contains the wrapped nodes."""

    def __init__(
        self,
        nodes: Iterable[Node | Pipeline],
        name: str,
        *,
        tags: str | Iterable[str] | None = None,
    ):
        self._name = name
        super().__init__(nodes, tags=tags)

    @property
    def nodes(self) -> list[Node]:
        return [FusedNode(self._nodes, name=self._name)]


class FusedRunner(SequentialRunner):
    """Fused runner is an extension of the SequentialRunner that
    essentially unpacks the FusedNode back to the contained nodes for
    execution."""

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str | None = None,
    ) -> None:
        nodes = pipeline.nodes
        super()._run(
            Pipeline([Pipeline(node._nodes) if isinstance(node, FusedNode) else node for node in nodes]),
            catalog,
            hook_manager,
            session_id,
        )
