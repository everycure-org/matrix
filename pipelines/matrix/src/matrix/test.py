# %%
import re
from pathlib import Path
from typing import Callable, Iterable, Any, Tuple

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline


project_path = "/Users/laurens/Documents/projects/matrix/pipelines/matrix"
metadata = bootstrap_project(project_path)
package_name = metadata.package_name

embeddings = pipelines["__default__"]

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
    def outputs(self) -> set[str]:
        return set().union(*[node.outputs for node in self._nodes])

    @property
    def tags(self) -> set[str]:
        return set().union(*[node.tags for node in self._nodes])
        

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


def clean_name(name: str) -> str:
    """Function to clean the node name.

    Args:
        name: name of the node
    Returns:
        Clean node name, according to Argo's requirements
    """
    return re.sub(r"[\W_]+", "-", name).strip("-")


nodes = {}

pipeline: Pipeline = pipelines["__default__"]
# for group in pipeline.grouped_nodes:
#     for node in group:
#         nodes[node] = node.outputs

def remove_params(elements):
    return [el for el in elements if not el.startswith("params:")]

fused = []
for group in pipeline.grouped_nodes:
    for target_node in group:
        # Find if there is a source node
        found = False

        if "argowf.fuse" in target_node.tags:
            for source_node in [fn for fn in fused if "argowf.fuse" in fn.tags]:
                if set(remove_params(target_node.inputs)) & set(source_node.outputs):
                    found = True
                    source_node.add_node(target_node)
        
        if not found:
            fused_node = FusedNode()
            fused_node.add_node(target_node)
            fused.append(fused_node)

# pipeline.grouped_nodes
[fuse.name for fuse in fused]

# %%
# dep_translation = {}
# fuse_node = None
# new_dependencies = {}
# for node, parents in embeddings.node_dependencies.items():
#     print("node", node.name)
#     print("parent", ",".join([parent.name for parent in parents]))
    
#     if "argowf.fuse" in node.tags and all(["argowf.fuse" in parent.tags for parent in parents]):
#         if fuse_node is None:
#             fuse_node = FusedNode()

#         fuse_node.add_node(node)
#         fuse_node.add_parents([parent if dep_translation.get(parent) is None else dep_translation[parent] for parent in parents if parent not in fuse_node._nodes])

#     else:
#         if fuse_node:

#             print("stopped fusing", node.name,  all(["argowf.fuse" in parent.tags for parent in parents]))

#             # add translated nodes
#             for fused_node in fuse_node._nodes:
#                 dep_translation[fused_node] = fuse_node

#             new_dependencies[fuse_node] = fuse_node._parents
#             fuse_node = None

#         new_dependencies[node] = [parent if dep_translation.get(parent) is None else dep_translation[parent] for parent in parents]

# if fuse_node:
#     new_dependencies[fuse_node] = fuse_node._parents

# d = [
#     {
#         "name":  clean_name(node.name),
#         # "name": clean_name(node.name),
#         "deps": [ clean_name(val.name) for val in sorted(parents)],
#         **{
#             tag.split("-")[0][len("argowf.") :]: tag.split("-")[1]
#             for tag in node.tags
#             if tag.startswith("argowf.") and "-" in tag
#         },
#     }
#     for node, parents in new_dependencies.items()
# ]

# sorted(d, key=lambda d: d["name"])
# %%
