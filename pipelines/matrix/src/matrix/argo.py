"""Module with utilities to generate Argo workflow."""

import re
from pathlib import Path
from typing import List, Optional, Any

import click
from jinja2 import Environment, FileSystemLoader

from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

TEMPLATE_FILE = "argo_wf_spec.tmpl"
RENDERED_FILE = "argo-workflow-template.yml"
SEARCH_PATH = Path("templates")


@click.group()
def cli() -> None:
    """Main CLI entrypoint."""
    ...


@click.command()
@click.argument("image", required=True)
@click.argument("image_tag", required=False, default="latest")
@click.argument("namespace", required=False, default="argo-workflows")
def generate_argo_config(
    image: str, run_name: str, image_tag: str, namespace: str, username: str
):
    """Function to render Argo pipeline template.

    Args:
        image: image to use
        run_name: name of the run
        image_tag: image tag to use
        namespace: the namespace in which to store the workflow
        pipeline_name: name of pipeline to generate
        env: execution environment
        username: user to execute
    """
    _generate_argo_config(image, run_name, image_tag, namespace, username)


def _generate_argo_config(
    image: str, run_name: str, image_tag: str, namespace: str, username: str
):
    loader = FileSystemLoader(searchpath=SEARCH_PATH)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template(TEMPLATE_FILE)

    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    package_name = metadata.package_name

    pipes = {}
    for name, pipeline in pipelines.items():
        # Fuse nodes in topological order to avoid constant recreation of Neo4j
        pipes[name] = get_dependencies(fuse(pipeline))

    output = template.render(
        package_name=package_name,
        pipes=pipes,
        image=image,
        image_tag=image_tag,
        namespace=namespace,
        username=username,
        run_name=run_name,
    )

    (SEARCH_PATH / RENDERED_FILE).write_text(output)


class FusedNode(Node):
    """Class to represent a fused node."""

    def __init__(  # noqa: PLR0913
        self, depth: int
    ):
        """Construct new instance of `FusedNode`."""
        self._nodes = []
        self._parents = set()
        self._inputs = []
        self.depth = depth

    def add_node(self, node):
        """Function to add node to group."""
        self._nodes.append(node)

    def add_parents(self, parents):
        """Function to set the parents of the group."""
        self._parents.update(set(parents))

    def fuses_with(self, node) -> bool:
        """Function verify fusability."""
        # If not is not fusable, abort
        if not self.is_fusable:
            return False

        # If fusing group does not match, abort
        if not self.fuse_group == self.get_fuse_group(node.tags):
            return False

        # Otherwise, fusable if connected
        return set(self.clean_dependencies(node.inputs)) & set(
            self.clean_dependencies(self.outputs)
        )

    @property
    def is_fusable(self) -> bool:
        """Check whether is fusable."""
        return "argowf.fuse" in self.tags

    @property
    def fuse_group(self) -> Optional[str]:
        """Retrieve fuse group."""
        return self.get_fuse_group(self.tags)

    @property
    def nodes(self) -> str:
        """Retrieve contained nodes."""
        return ",".join([node.name for node in self._nodes])

    @property
    def outputs(self) -> set[str]:
        """Retrieve output datasets."""
        return set().union(
            *[self.clean_dependencies(node.outputs) for node in self._nodes]
        )

    @property
    def tags(self) -> set[str]:
        """Retrieve tags."""
        return set().union(*[node.tags for node in self._nodes])

    @property
    def name(self) -> str:
        """Retrieve name of fusedNode."""
        if self.is_fusable and len(self._nodes) > 1:
            return self.fuse_group

        # If not fusable, revert to name of node
        return self._nodes[0].name

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

    @staticmethod
    def get_fuse_group(tags: str) -> Optional[str]:
        """Function to retrieve fuse group."""
        for tag in tags:
            if tag.startswith("argowf.fuse-group."):
                return tag[len("argowf.fuse-group.") :]

        return None

    @staticmethod
    def clean_dependencies(elements) -> List[str]:
        """Function to clean node dependencies.

        Operates by removing params: from the list and dismissing
        the transcoding operator.
        """
        return [el.split("@")[0] for el in elements if not el.startswith("params:")]


def fuse(pipeline: Pipeline) -> List[FusedNode]:
    """Function to fuse given pipeline.

    Leverages the Tags provided by Kedro to fuse nodes for execution
    by a single Argo Workflow step.

    Args:
        pipeline: Kedro pipeline
    Returns
        List of fusedNodes with their dependencies
    """
    fused = []

    # Kedro provides the `grouped_nodes` property, that yields a list of node groups that can
    # be executed in topological order. We're using this as the starting point for our fusing algorithm.
    for depth, group in enumerate(pipeline.grouped_nodes):
        for target_node in group:
            # Find source node that provides its inputs
            num_fused = 0
            fuse_node = None

            # Given a topological node, we're trying to find a parent node
            # to which it can be fused. Nodes can be fused with they have the
            # proper labels and they have dataset dependencies, and the parent
            # is in the previous node group.
            for source_node in fused:
                if (
                    source_node.fuses_with(target_node)
                    and source_node.depth == depth - 1
                ):
                    fuse_node = source_node
                    num_fused = num_fused + 1

            # We only fuse if there is a single parent to fuse with
            # if multiple parents, avoid fusing otherwise this might
            # mess with dependencies.
            if num_fused == 1:
                fuse_node.depth = depth
                fuse_node.add_node(target_node)
                fuse_node.add_parents(
                    [
                        fs
                        for fs in fused
                        if set(FusedNode.clean_dependencies(target_node.inputs))
                        & set(FusedNode.clean_dependencies(fs.outputs))
                        if fs != fuse_node
                    ]
                )

            # If we can't find any nodes to fuse to, we're adding this node
            # as an independent node to the result, which implies it will be executed
            # using it's own Argo node unless a downstream node will be fused to it.
            else:
                fused_node = FusedNode(depth)
                fused_node.add_node(target_node)
                fused_node.add_parents(
                    [
                        fs
                        for fs in fused
                        if set(FusedNode.clean_dependencies(target_node.inputs))
                        & set(FusedNode.clean_dependencies(fs.outputs))
                    ]
                )
                fused.append(fused_node)

    return fused


def get_dependencies(fused_pipeline: List[FusedNode]):
    """Function to yield node dependencies to render Argo template.

    Args:
        fused_pipeline: fused pipeline
    Return:
        Dictionary to render Argo template
    """
    deps_dict = [
        {
            "name": clean_name(fuse.name),
            "nodes": fuse.nodes,
            "deps": [clean_name(val.name) for val in sorted(fuse._parents)],
            "tags": fuse.tags,
            **{
                tag.split("-")[0][len("argowf.") :]: tag.split("-")[1]
                for tag in fuse.tags
                if tag.startswith("argowf.") and "-" in tag
            },
        }
        for fuse in fused_pipeline
    ]
    return sorted(deps_dict, key=lambda d: d["name"])


def clean_name(name: str) -> str:
    """Function to clean the node name.

    Args:
        name: name of the node
    Returns:
        Clean node name, according to Argo's requirements
    """
    return re.sub(r"[\W_]+", "-", name).strip("-")


if __name__ == "__main__":
    cli.add_command(generate_argo_config)
    cli()
