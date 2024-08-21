"""Module with utilities to generate Argo workflow."""
import re
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader

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


# Validate if nodes can be fused in topological order
def is_fusable(pipeline):
    # NOTE: Currently a pipeline is fusable, if all it's nodes have the `argo-wf.fuse` label.


    if len(pipeline._nodes) == len(
        pipeline.only_nodes_with_tags("argo-wf.fuse")._nodes
    ):
        return True

    return False


def fuse(pipeline):

    # NOTE: Lol if you don't call pipeline, topological groups is empty?
    print(pipeline)

    for node in pipeline.nodes


def get_dependencies(fused_pipeline):
    """Function to yield node dependencies to render Argo template.

    Args:
        dependencies: pipeline dependencies
    Return:
        Dictionary to render Argo template
    """
    deps_dict = [
        {
            "nodes": [node.name for node in group]
            # "name": clean_name(node.name),
            # "deps": [clean_name(val.name) for val in sorted(parent_nodes)],
            # **{
            #     tag.split("-")[0][len("argo.") :]: tag.split("-")[1]
            #     for tag in node.tags
            #     if tag.startswith("argo.")
            # },
        }
        for group in fused_pipeline
    ]
    return sorted(deps_dict, key=lambda d: d["nodes"])


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
