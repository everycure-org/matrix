"""Module with utilities to generate Argo workflow."""
import re
from pathlib import Path
from typing import Callable, Iterable, Any, Tuple

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
def generate_argo_config(image, image_tag):
    """Function to render Argo pipeline template.

    Args:
        image: image to use
        image_tag: image tag to use
        pipeline_name: name of pipeline to generate
        env: execution environment
    """
    loader = FileSystemLoader(searchpath=SEARCH_PATH)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template(TEMPLATE_FILE)

    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    package_name = metadata.package_name

    pipes = {}
    for name, pipeline in pipelines.items():
        # TODO: Fuse nodes in topological order to avoid constant recreation of Neo4j
        pipes[name] = get_dependencies(fuse(pipeline))

    output = template.render(
        package_name=package_name, pipes=pipes, image=image, image_tag=image_tag
    )

    (SEARCH_PATH / RENDERED_FILE).write_text(output)


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

    def fuses_with(self, node):
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
    def is_fusable(self):
        return "argowf.fuse" in self.tags

    @property
    def fuse_group(self):
        return self.get_fuse_group(self.tags)

    @property
    def nodes(self) -> str:
        return ",".join([node.name for node in self._nodes])

    @property
    def outputs(self) -> set[str]:
        return set().union(*[node.outputs for node in self._nodes])

    @property
    def tags(self) -> set[str]:
        return set().union(*[node.tags for node in self._nodes])

    @property
    def name(self):
        if self.is_fusable:
            return self.fuse_group

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
    def get_fuse_group(tags):
        for tag in tags:
            if tag.startswith("argowf.fuse-group."):
                return tag[len("argowf.fuse-group.") :]

        return None

    @staticmethod
    def remove_transcoding(dataset: str):
        return dataset.split("@")[0]

    @staticmethod
    def clean_dependencies(elements):
        # Remove params and remove transcoding
        return [
            FusedNode.remove_transcoding(el)
            for el in elements
            if not el.startswith("params:")
        ]


def fuse(pipeline: Pipeline):

    fused = []
    
    for group in pipeline.grouped_nodes:
        for target_node in group:
            # Find source node that provides its inputs
            found = False

            for source_node in fused:
                if source_node.fuses_with(target_node):
                    found = True
                    source_node.add_node(target_node)

            if not found:
                fused_node = FusedNode()
                fused_node.add_node(target_node)
                fused_node.set_parents(
                    [
                        fs
                        for fs in fused
                        if set(FusedNode.clean_dependencies(target_node.inputs))
                        & set(FusedNode.clean_dependencies(fs.outputs))
                    ]
                )
                fused.append(fused_node)

    return fused


def get_dependencies(fused_pipeline):
    """Function to yield node dependencies to render Argo template.

    Args:
        dependencies: pipeline dependencies
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
