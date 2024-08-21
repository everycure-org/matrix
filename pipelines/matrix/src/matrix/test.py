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

    def set_parents(self, parents):
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
            for source_node in [fs for fs in fused if "argowf.fuse" in fs.tags]:
                if set(remove_params(target_node.inputs)) & set(source_node.outputs):
                    found = True
                    source_node.add_node(target_node)

        if not found:
            fused_node = FusedNode()
            fused_node.add_node(target_node)
            fused_node.set_parents(
                [
                    fs
                    for fs in fused
                    if set(remove_params(target_node.inputs)) & set(fs.outputs)
                ]
            )
            fused.append(fused_node)

# pipeline.grouped_nodes
{fuse.name: {"deps": [f.name for f in fuse._parents]} for fuse in fused}

d = [
    {
        "name": clean_name(fuse.name),
        # "name": clean_name(node.name),
        "deps": [clean_name(val.name) for val in sorted(fuse._parents)],
        "tags": fuse.tags,
        **{
            tag.split("-")[0][len("argowf.") :]: tag.split("-")[1]
            for tag in fuse.tags
            if tag.startswith("argowf.") and "-" in tag
        },
    }
    for fuse in fused
]

sorted(d, key=lambda d: d["name"])
# %%
