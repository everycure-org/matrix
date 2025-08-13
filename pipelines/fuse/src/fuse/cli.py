import re
from pathlib import Path
from typing import Any, Dict, List

import click
import yaml
from jinja2 import Environment, FileSystemLoader
from kedro.framework.cli.utils import CONTEXT_SETTINGS, KedroCliError
from kedro.framework.project import pipelines as kedro_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node

ARGO_TEMPLATES_DIR_PATH = Path(__file__).parent.parent.parent / "templates"


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    pass


@cli.command()
@click.option("--pipeline", "-p", type=str, default="modelling_run", help="Specify which pipeline to execute")
@click.pass_context
def compile(
    ctx,
    pipeline: str,
):
    loader = FileSystemLoader(searchpath=ARGO_TEMPLATES_DIR_PATH)
    template_env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = template_env.get_template("argo_wf_spec.tmpl")
    pipeline_tasks = get_argo_dag(kedro_pipelines[pipeline])

    # Render the template
    rendered_template = template.render(
        pipeline_tasks=[task.to_dict() for task in pipeline_tasks.values()],
        pipeline_name=pipeline,
    )

    yaml_data = yaml.safe_load(rendered_template)
    yaml_without_anchors = yaml.dump(yaml_data, sort_keys=False, default_flow_style=False)
    save_argo_template(
        yaml_without_anchors,
    )


class ArgoTask:
    """Class to model an Argo task.

    Argo's operating model slightly differs from Kedro's, i.e., while Kedro uses dataset
    dependecies to model relationships, Argo uses task dependencies."""

    def __init__(self, node: Node):
        self._node = node
        self._parents = []

    @property
    def node(self):
        return self._node

    def add_parents(self, nodes: List[Node]):
        self._parents.extend(nodes)

    def to_dict(self):
        return {
            "name": clean_name(self._node.name),
            "nodes": self._node.name,
            "deps": [clean_name(parent.name) for parent in sorted(self._parents)],
        }


def get_argo_dag(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """Function to convert the Kedro pipeline into Argo Tasks. The function
    iterates the nodes of the pipeline and generates Argo tasks with dependencies.
    These dependencies are inferred based on the input and output datasets for
    each node.

    NOTE: This function is now agnostic to the fact that nodes might be fused. The nodes
    returned as part of the pipeline may optionally contain FusedNodes, which have correct
    inputs and outputs for the perspective of the Argo Task.
    """
    tasks = {}

    # The `grouped_nodes` property returns the nodes list, in a toplogical order,
    # allowing us to easily translate the Kedro DAG to an Argo WF.
    for group in pipeline.grouped_nodes:
        for target_node in group:
            task = ArgoTask(target_node)
            task.add_parents(
                [
                    parent.node
                    for parent in tasks.values()
                    if set(clean_dependencies(target_node.inputs)) & set(clean_dependencies(parent.node.outputs))
                ]
            )

            tasks[target_node.name] = task

    return tasks


def clean_name(name: str) -> str:
    """Function to clean the node name.

    Args:
        name: name of the node
    Returns:
        Clean node name, according to Argo's requirements
    """
    return re.sub(r"[\W_]+", "-", name).strip("-")


def clean_dependencies(elements) -> List[str]:
    """Function to clean node dependencies.

    Operates by removing params: from the list and dismissing
    the transcoding operator.
    """
    return [el.split("@")[0] for el in elements if not el.startswith("params:")]


if __name__ == "__main__":
    try:
        cli()
    except KedroCliError as e:
        raise e
    except Exception as e:
        raise KedroCliError(str(e))


def save_argo_template(argo_template: str) -> str:
    file_path = ARGO_TEMPLATES_DIR_PATH / "argo-workflow-template.yml"
    with open(file_path, "w") as f:
        f.write(argo_template)
    return str(file_path)
