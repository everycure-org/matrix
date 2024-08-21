# %%
from pathlib import Path
from typing import Callable, Iterable, Any, Tuple

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project
from kedro.pipeline.node import Node


project_path = "/Users/laurens/Documents/projects/matrix/pipelines/matrix"
metadata = bootstrap_project(project_path)
package_name = metadata.package_name

embeddings = pipelines["embeddings"]

class FusedNode(Node):

    def __init__(  # noqa: PLR0913
        self,
    ):
        self._nodes = []
        self._parents = set()
        self._inputs = []

    def add_node(self, node):
        self._nodes.append(node)

    def add_parents(self, parents):
        self._parents.update(parents)

    @property
    def tags(self) -> set[str]:
        # TODO

    @property
    def name(self):
        return ",".join([node.name for node in self._nodes])
    
    @property
    def _unique_key(self) -> tuple[Any, Any] | Any | tuple:
        def hashable(value: Any) -> tuple[Any, Any] | Any | tuple:
            if isinstance(value, dict):
                # we sort it because a node with inputs/outputs
                # {"arg1": "a", "arg2": "b"} is equivalent to
                # a node with inputs/outputs {"arg2": "b", "arg1": "a"}
                return tuple(sorted(value.items()))
            if isinstance(value, list):
                return tuple(value)
            return value

        return self.name, hashable(self._nodes)


dep_translation = {}
fuse_node = None
new_dependencies = {}
for node, parents in embeddings.node_dependencies.items():
    print("node", node.name)
    print("parent", ",".join([parent.name for parent in parents]))
    
    if "argo-wf.fuse" in node.tags and all(["argo-wf.fuse" in parent.tags for parent in parents]):
        if fuse_node is None:
            fuse_node = FusedNode()

        fuse_node.add_node(node)
        fuse_node.add_parents([parent if dep_translation.get(parent) is None else dep_translation[parent] for parent in parents])

    else:
        if fuse_node:

            # add translated nodes
            for node in fuse_node._nodes:
                dep_translation[node] = fuse_node

            new_dependencies[fuse_node] = fuse_node._parents
            fuse_node = None

        new_dependencies[node] = [parent if dep_translation.get(parent) is None else dep_translation[parent] for parent in parents]

if fuse_node:
    new_dependencies[fuse_node] = fuse_node._parents


d = [
    {
        "name": node.name
        # "name": clean_name(node.name),
        # "deps": [clean_name(val.name) for val in sorted(parent_nodes)],
        # **{
        #     tag.split("-")[0][len("argo.") :]: tag.split("-")[1]
        #     for tag in node.tags
        #     if tag.startswith("argo.")
        # },
    }
    for node, parents in new_dependencies.items()
]

d

# %%
