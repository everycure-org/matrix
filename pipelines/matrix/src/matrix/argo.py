"""Module with utilities to generate Argo workflow."""
import re
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

TEMPLATE_FILE = "argo_spec.tmpl"
SEARCH_PATH = Path("templates")


@click.command()
@click.argument("image", required=True)
@click.option("-p", "--pipeline", "pipeline_name", default=None)
@click.option("--env", "-e", type=str, default="base")
def generate_argo_config(image, pipeline_name, env):
    """Function to render Argo pipeline template.

    Args:
        image: image to use
        pipeline_name: name of pipeline to generate
        env: execution environment
    """
    loader = FileSystemLoader(searchpath=SEARCH_PATH)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template(TEMPLATE_FILE)

    project_path = Path.cwd()
    metadata = bootstrap_project(project_path)
    package_name = metadata.package_name

    pipeline_name = pipeline_name or "__default__"
    pipeline = pipelines.get(pipeline_name)

    tasks = get_dependencies(pipeline.node_dependencies)

    output = template.render(
        image=image,
        package_name=package_name,
        tasks=tasks,
        pipeline=pipeline_name,
        env=env,
    )

    (SEARCH_PATH / f"argo-{package_name}.yml").write_text(output)


def get_dependencies(dependencies):
    """Function to yield node dependencies to render Argo template.

    Args:
        dependencies: pipeline dependencies
    Return:
        Dictionary to render Argo template
    """
    deps_dict = [
        {
            "node": node.name,
            "name": clean_name(node.name),
            "deps": [clean_name(val.name) for val in parent_nodes],
            **{
                tag.split("-")[0][len("argo.") :]: tag.split("-")[1]
                for tag in node.tags
                if tag.startswith("argo.")
            },
        }
        for node, parent_nodes in dependencies.items()
    ]
    return deps_dict


def clean_name(name: str) -> str:
    """Function to clean the node name.

    Args:
        name: name of the node
    Returns:
        Clean node name, according to Argo's requirements
    """
    return re.sub(r"[\W_]+", "-", name).strip("-")


if __name__ == "__main__":
    generate_argo_config()
