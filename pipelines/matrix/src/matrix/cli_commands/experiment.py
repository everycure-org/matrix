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

from matrix.cli_commands.run import run
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
@click.argument( "function_to_call",      type=str, default=None,)
# @click.option( "--name",   type=str, default="")
# @click.option( "--from-inputs",   type=str, default="", help=FROM_INPUTS_HELP, callback=split_string)
# @click.option( "--to-outputs",    type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string)
# @click.option( "--from-nodes",    type=str, default="", help=FROM_NODES_HELP, callback=split_node_names,)
# @click.option( "--to-nodes",      type=str, default="", help=TO_NODES_HELP, callback=split_node_names)
@click.option( "--nodes",         "-n", "node_names", type=str, multiple=False, help=NODE_ARG_HELP, callback=split_string, default="",)
# @click.option( "--runner",        "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP)
# @click.option("--async",          "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP) 
# @click.option("--tags",           "-t", type=str, multiple=True, help=TAG_ARG_HELP)
# @click.option( "--without-tags",  "-wt", type=str, help="used to filter out nodes with tags that should not be run. All dependent downstream nodes are also removed. Note nodes need to have _all_ tags to be removed.", callback=split_string, default=[],)
# @click.option( "--load-versions", "-lv", type=str, multiple=True, help=LOAD_VERSION_HELP, callback=_split_load_versions,)
# @click.option("--pipeline",       "-p", required=True, default="__default__", type=str, help=PIPELINE_ARG_HELP)
# @click.option( "--conf-source",   type=click.Path(exists=True, file_okay=False, resolve_path=True), help=CONF_SOURCE_HELP,)
# @click.option( "--params",        type=click.UNPROCESSED, default="", help=PARAMS_ARG_HELP, callback=_split_params,)
# @click.option( "--from-env",      type=str, default=None, help="Custom env to read from, if specified will read from the `--from-env` and write to the `--env`",)
@click.option( "--experiment_name",      type=str, default=None,)
# fmt: on
# def experiment(env:str, name: Optional[str]):
# def experiment(tags: list[str], without_tags: list[str], env:str, runner: str, is_async: bool, node_names: list[str], to_nodes: list[str], from_nodes: list[str], from_inputs: list[str], to_outputs: list[str], load_versions: list[str], pipeline: str, conf_source: str, params: dict[str, Any], from_env: Optional[str]=None, experiment_id: Optional[str]=None):
@click.pass_context
def experiment(ctx, function_to_call, env:str, node_names: list[str], experiment_name: Optional[str]):
    """Run an experiment."""
    
    if not experiment_name:
        # TODO: sanitize branch name
        experiment_name = get_current_git_branch()
        print("experiment_name", experiment_name)

    run_id = get_run_id_from_mlflow(experiment_name=experiment_name)

    if function_to_call == "run":
        # invokes another command with the arguments you provide as a caller
        ctx.invoke(run, experiment_id=run_id, node_names=node_names)
    elif  function_to_call == "submit":
        # TODO add submit
        print("Run kedro submit")
    else:
        print(f"{function_to_call} not a valid option")
        raise click.Abort()



def get_run_id_from_mlflow(experiment_name:str):

    token = get_iap_token()
    # print(token.to_json())

    # mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
    mlflow.set_tracking_uri("http://127.0.0.1:5001/")
    os.environ["MLFLOW_TRACKING_TOKEN"] = token.id_token

    # try: 
    experiment = mlflow.get_experiment_by_name(name=experiment_name)
    # except 

    print("experiment", experiment)

    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Experiment {experiment_name} found with ID {experiment_id}")
    else:
        print(f"Experiment {experiment_name} not found. Generating...")
        # TODO: Add error handling
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Experiment {experiment_name} generated with ID {experiment_id}")

    return experiment_id
