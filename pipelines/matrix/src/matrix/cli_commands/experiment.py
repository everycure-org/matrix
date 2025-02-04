import os
from typing import Any, Collection, Dict, List, NamedTuple, Optional, Set

import click
import mlflow
from kedro.framework.cli.project import (
    ASYNC_ARG_HELP,
    CONF_SOURCE_HELP,
    FROM_INPUTS_HELP,
    FROM_NODES_HELP,
    LOAD_VERSION_HELP,
    NODE_ARG_HELP,
    PARAMS_ARG_HELP,
    PIPELINE_ARG_HELP,
    RUNNER_ARG_HELP,
    TAG_ARG_HELP,
    TO_NODES_HELP,
    TO_OUTPUTS_HELP,
    project_group,
)
from kedro.framework.cli.utils import (
    _split_load_versions,
    _split_params,
    env_option,
    split_node_names,
    split_string,
)
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.framework.project import pipelines, settings
from kedro.io import DataCatalog
from kedro.pipeline.pipeline import Pipeline
from kedro.utils import load_obj

from matrix.git_utils import get_current_git_branch
from matrix.session import KedroSessionWithFromCatalog
from matrix.utils.authentication import get_iap_token


class RunConfig(NamedTuple):
    # pipeline_obj: Optional[Pipeline]
    # pipeline_name: str
    # env: str
    # runner: str
    # is_async: bool
    # node_names: List[str]
    # to_nodes: List[str]
    # from_nodes: List[str]
    # from_inputs: List[str]
    # to_outputs: List[str]
    # load_versions: dict[str, str]
    # tags: List[str]
    # without_tags: List[str]
    # conf_source: Optional[str]
    # params: Dict[str, Any]
    # from_env: Optional[str]
    name: str


# fmt: off
@project_group.command()
@env_option
@click.option( "--name",   type=str, default="")
# fmt: on
def experiment(env:str, name: Optional[str]):
    """Create a new experiment."""
    config = RunConfig(
        name=name
    )

    print("here")
    print('user provided name', name)

    _experiment(name)


def _experiment(name: str) -> None:
    print("Experiment name", name)

    if not name:
        name = get_current_git_branch()

    print('new nme', name)

    run_id = get_run_id_from_mlflow(experiment_name=name)

    print(run_id)



def get_run_id_from_mlflow(experiment_name:str):

    token = get_iap_token()
    # print(token.to_json())

    mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token

    # try: 
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    # except 

    print(experiment)

    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Experiment {experiment_name} found with ID {experiment_id}")
    else:
        print(f"Experiment {experiment_name} not found. Generating...")
        # TODO: Add error handling
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Experiment {experiment_name} generated with ID {experiment_id}")

    return experiment_id
