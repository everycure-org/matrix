import os
from typing import Any, Collection, Dict, List, NamedTuple, Optional, Set

import click
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

from matrix.utils.environment import load_environment_variables

# Load environment variables from .env.defaults and .env
load_environment_variables()

from matrix.session import KedroSessionWithFromCatalog


class RunConfig(NamedTuple):
    pipeline_obj: Optional[Pipeline]
    pipeline_name: str
    env: str
    runner: str
    is_async: bool
    node_names: List[str]
    to_nodes: List[str]
    from_nodes: List[str]
    from_inputs: List[str]
    to_outputs: List[str]
    load_versions: dict[str, str]
    tags: List[str]
    without_tags: List[str]
    conf_source: Optional[str]
    params: Dict[str, Any]
    from_env: Optional[str]
    from_run: Optional[str]


# fmt: off
@project_group.command()
@env_option
@click.option( "--from-inputs",   type=str, default="", help=FROM_INPUTS_HELP, callback=split_string)
@click.option( "--to-outputs",    type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string)
@click.option( "--from-nodes",    type=str, default="", help=FROM_NODES_HELP, callback=split_node_names,)
@click.option( "--to-nodes",      type=str, default="", help=TO_NODES_HELP, callback=split_node_names)
@click.option( "--nodes",         "-n", "node_names", type=str, multiple=False, help=NODE_ARG_HELP, callback=split_string, default="",)
@click.option( "--runner",        "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP)
@click.option("--async",          "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP) 
@click.option("--tags",           "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option( "--without-tags",  "-wt", type=str, help="used to filter out nodes with tags that should not be run. All dependent downstream nodes are also removed. Note nodes need to have _all_ tags to be removed.", callback=split_string, default=[],)
@click.option( "--load-versions", "-lv", type=str, multiple=True, help=LOAD_VERSION_HELP, callback=_split_load_versions,)
@click.option("--pipeline",       "-p", required=True, default="__default__", type=str, help=PIPELINE_ARG_HELP)
@click.option( "--conf-source",   type=click.Path(exists=True, file_okay=False, resolve_path=True), help=CONF_SOURCE_HELP,)
@click.option( "--params",        type=click.UNPROCESSED, default="", help=PARAMS_ARG_HELP, callback=_split_params,)
@click.option( "--from-env",      type=str, default=None, help="Custom env to read from, if specified will read from the `--from-env` and write to the `--env`",)
@click.option( "--from-run",      type=str, default=None, help="Custom run to read from, if specified will read all input datasets from the `--from-run` and write outputs to the current run",)
# fmt: on
def run(tags: list[str], without_tags: list[str], env:str, runner: str, is_async: bool, node_names: list[str], to_nodes: list[str], from_nodes: list[str], from_inputs: list[str], to_outputs: list[str], load_versions: list[str], pipeline: str, conf_source: str, params: dict[str, Any], from_env: Optional[str]=None, from_run: Optional[str]=None):
    """Run the pipeline."""

    _validate_env_vars_for_private_data()

    pipeline_name = pipeline
    pipeline_obj = pipelines[pipeline_name]

    config = RunConfig(
        pipeline_obj=pipeline_obj,
        pipeline_name=pipeline_name,
        env=env,
        runner=runner,
        is_async=is_async,
        node_names=node_names,
        to_nodes=to_nodes,
        from_nodes=from_nodes,
        from_inputs=from_inputs,
        to_outputs=to_outputs,
        load_versions=load_versions,
        tags=tags,
        without_tags=without_tags,
        conf_source=conf_source,
        params=params,
        from_env=from_env,
        from_run=from_run
    )

    _run(config, KedroSessionWithFromCatalog)


def _validate_env_vars_for_private_data() -> None:
    """ Short-circuit if a user is requesting private datasets but their project or bucket point to dev. """
    env_vars = {
        "RUNTIME_GCP_PROJECT_ID": os.environ["RUNTIME_GCP_PROJECT_ID"],
        "RUNTIME_GCP_BUCKET": os.environ["RUNTIME_GCP_BUCKET"],
        "MLFLOW_URL": os.environ["MLFLOW_URL"],
    }

    if os.getenv("INCLUDE_PRIVATE_DATASETS", "") == "1":
        for var_name, var_value in env_vars.items():
            if "prod" not in var_value.lower() or "dev" in var_value.lower():
                click.confirm(
                    f"You requested private datasets, yet the value of env var {var_name} - {var_value } does not point to a production resource. "
                    f"Doing so runs the risk of exposing private datasets to the public, which can have legal consequences. "
                    f"Are you aware of this, and want to continue with your current settings nevertheless?",
                    abort=True,
                    err=True
                )


def _run(config: RunConfig, kedro_session: KedroSessionWithFromCatalog) -> None:
    if config.pipeline_name in ["test", "fabricator"] and config.env in [None, "base"]:
        raise RuntimeError(
            "Running the fabricator in the base environment might overwrite production data! Use the test env `-e test` instead."
        )
    elif config.pipeline_name in ["create_sample", "test_sample"]  and config.env not in ["sample"]:
        raise RuntimeError(
            "Running the sample pipelines outside of the sample environment might overwrite production data! Use the sample env `-e sample` instead."
            )

    runner = load_obj(config.runner or "SequentialRunner", "kedro.runner")

    with kedro_session.create(
        env=config.env, conf_source=config.conf_source, extra_params=config.params
    ) as session:
        # introduced to filter out tags that should not be run
        node_names = _filter_nodes_missing_tag(
            without_tags=config.without_tags, pipeline_obj=config.pipeline_obj, node_names=config.node_names
        )

        from_catalog = _extract_config(config, session)
        
        # If using --from-run, we want to use the from_catalog for inputs
        # but still write outputs to the current run
        if config.from_run:
            # Use from_catalog for all input datasets
            # This will make Kedro read inputs from the from_catalog
            # while writing outputs to the current run
            print(f"Using from_catalog for inputs from run: {config.from_run}")
            print(f"Outputs will be written to current run")
        
        session.run(
            from_catalog=from_catalog,
            tags=config.tags,
            runner=runner(is_async=config.is_async),
            node_names=node_names,
            from_nodes=config.from_nodes,
            to_nodes=config.to_nodes,
            from_inputs=config.from_inputs,
            to_outputs=config.to_outputs,
            load_versions=config.load_versions,
            pipeline_name=config.pipeline_name,
        )


def _extract_config(config: RunConfig, session: KedroSessionWithFromCatalog) -> Optional[DataCatalog]:
    from_catalog: Optional[DataCatalog] = None
    
    if config.from_env:
        # Load second config loader instance for from-env
        config_loader_class = settings.CONFIG_LOADER_CLASS
        config_loader = config_loader_class(  # type: ignore[no-any-return]
                conf_source=session._conf_source,
                env=config.from_env,
                **settings.CONFIG_LOADER_ARGS,
            )
        conf_catalog = config_loader["catalog"]
        conf_catalog = _convert_paths_to_absolute_posix(
                project_path=session._project_path, conf_dictionary=conf_catalog
            )
        conf_creds = config_loader["credentials"]
        from_catalog: DataCatalog = settings.DATA_CATALOG_CLASS.from_config(
                catalog=conf_catalog, credentials=conf_creds
            )
        from_catalog.add_feed_dict(_get_feed_dict(config_loader["parameters"]), replace=True)
    
    elif config.from_run:
        # Load config for from-run: redirect input datasets to specified run
        from_catalog = _create_from_run_catalog(config.from_run, config.env, session, config.pipeline_name)
    
    return from_catalog


def _create_from_run_catalog(from_run: str, env: str, session: KedroSessionWithFromCatalog, pipeline_name: str) -> DataCatalog:
    """Create a catalog that contains datasets from the specified run.
    
    Args:
        from_run: The run name to read input datasets from
        env: The current environment
        session: The Kedro session
        pipeline_name: The pipeline name being run
        
    Returns:
        DataCatalog with datasets from the specified run
    """
    # NOTE: This function was partially generated using AI assistance.
    
    print(f"Creating from-run catalog for run: {from_run}")
    
    # Load the current environment's catalog configuration
    config_loader_class = settings.CONFIG_LOADER_CLASS
    config_loader = config_loader_class(
        conf_source=session._conf_source,
        env=session.store.get("env", "base"),
        **settings.CONFIG_LOADER_ARGS,
    )
    
    # Get the catalog and credentials
    conf_catalog = config_loader["catalog"]
    conf_creds = config_loader["credentials"]
    
    # Create a new catalog from the current environment's config
    from_catalog: DataCatalog = settings.DATA_CATALOG_CLASS.from_config(
        catalog=conf_catalog, credentials=conf_creds
    )
    
    # Get the pipeline to identify which datasets are inputs
    pipeline = pipelines[pipeline_name]
    
    # Get the filtered pipeline based on the run configuration
    # This will help us identify which datasets are actually needed
    filtered_pipeline = pipeline.filter(
        tags=config_loader.get("tags", []),
        from_nodes=config_loader.get("from_nodes", []),
        to_nodes=config_loader.get("to_nodes", []),
        node_names=config_loader.get("node_names", []),
        from_inputs=config_loader.get("from_inputs", []),
        to_outputs=config_loader.get("to_outputs", []),
    )
    
    # Get all input datasets that will be used
    input_datasets = filtered_pipeline.inputs()
    # print(f"Input datasets that will be used: {input_datasets}")
    # print(f"Available datasets in catalog: {list(from_catalog._datasets.keys())}")
    
    # Filter to only datasets that actually exist in the catalog
    # (some inputs are outputs from other pipelines that don't exist yet)
    existing_input_datasets = [ds for ds in input_datasets if ds in from_catalog._datasets]
    print(f"Existing input datasets in catalog: {existing_input_datasets}")
    
    # Filter out parameter datasets (those starting with 'params:')
    # Parameters are not catalog entries and should not be redirected
    data_input_datasets = [ds for ds in existing_input_datasets if not ds.startswith('params:')]
    print(f"Data input datasets to redirect: {data_input_datasets}")
    breakpoint()
    # Redirect input datasets to the specified run
    for dataset_name in data_input_datasets:
        print(f"Checking dataset: {dataset_name}")
        dataset = from_catalog._datasets[dataset_name]
        print(f"Found dataset {dataset_name} with filepath: {getattr(dataset, '_filepath', 'No filepath')}")
        breakpoint()
        # Only redirect datasets that have filepaths (skip parameters and cache datasets)
        if hasattr(dataset, '_filepath') and dataset._filepath:
            # Convert PurePosixPath to string for string operations
            filepath_str = str(dataset._filepath)
            # Check if this dataset uses a run-based path
            if '/runs/' in filepath_str and '/datasets/' in filepath_str:
                # Redirect the filepath to the specified run
                redirected_filepath = _redirect_filepath_to_run(filepath_str, from_run, env)
                if redirected_filepath != filepath_str:
                    print(f"Redirecting {dataset_name} from {filepath_str} to {redirected_filepath}")
                    dataset._filepath = redirected_filepath
                breakpoint()
            else:
                print(f"Skipping {dataset_name} - not a run-based dataset")
        else:
            print(f"Skipping {dataset_name} - no filepath (likely parameter or cache dataset)")
    
    return from_catalog


def _redirect_filepath_to_run(filepath: str, from_run: str, env: str) -> str:
    """Redirect a filepath to use the specified run.
    
    Args:
        filepath: The original filepath
        from_run: The run name to redirect to
        env: The current environment
        
    Returns:
        The redirected filepath
    """
    # NOTE: This function was partially generated using AI assistance.
    
    # Check if this is a run-based path that should be redirected
    if env == "test":
        # For test environment, redirect run-based paths
        if "/runs/" in filepath and "/datasets/" in filepath:
            # Extract the path components
            parts = filepath.split("/")
            run_index = parts.index("runs")
            datasets_index = parts.index("datasets")
            
            if run_index < datasets_index:
                # Replace the run name in the path
                parts[run_index + 1] = from_run
                return "/".join(parts)
    
    elif env == "cloud":
        # For cloud environment, redirect run-based paths
        if "/runs/" in filepath and "/datasets/" in filepath:
            # Extract the path components
            parts = filepath.split("/")
            run_index = parts.index("runs")
            datasets_index = parts.index("datasets")
            
            if run_index < datasets_index:
                # Replace the run name in the path
                parts[run_index + 1] = from_run
                return "/".join(parts)
    
    # Return original filepath if no redirection needed
    return filepath


def _get_feed_dict(params: Dict) -> dict[str, Any]:
    """Get parameters and return the feed dictionary."""
    feed_dict = {"parameters": params}

    def _add_param_to_feed_dict(param_name: str, param_value: Any) -> None:
        """Add param to feed dict.

        This recursively adds parameter paths to the `feed_dict`,
        whenever `param_value` is a dictionary itself, so that users can
        specify specific nested parameters in their node inputs.

        Example:
            >>> param_name = "a"
            >>> param_value = {"b": 1}
            >>> _add_param_to_feed_dict(param_name, param_value)
            >>> assert feed_dict["params:a"] == {"b": 1}
            >>> assert feed_dict["params:a.b"] == 1
        """
        key = f"params:{param_name}"
        feed_dict[key] = param_value
        if isinstance(param_value, dict):
            for key, val in param_value.items():
                _add_param_to_feed_dict(f"{param_name}.{key}", val)

    for param_name, param_value in params.items():
        _add_param_to_feed_dict(param_name, param_value)

    return feed_dict


def _filter_nodes_missing_tag(
    without_tags: Collection[str], pipeline_obj: Pipeline, node_names: Collection[str]
) -> set[str]:
    """Filter out nodes that have tags that should not be run and their downstream nodes."""
    if not without_tags:
        return set(node_names)

    without_tags: Set[str] = set(without_tags)

    if not node_names:
        node_names = {node.name for node in pipeline_obj.nodes}

    # Step 1: Identify nodes to remove
    nodes_to_remove = set(
        node.name for node in pipeline_obj.nodes if node.tags.issuperset(without_tags)
    )

    # Step 2: Identify and add downstream nodes
    downstream_nodes = pipeline_obj.from_nodes(*nodes_to_remove).nodes
    ds_nodes_names = [node.name for node in downstream_nodes]

    nodes_to_remove.update(ds_nodes_names)

    # Step 3: Filter the node_names
    filtered_nodes = set(node_names).difference(nodes_to_remove)

    # Step 4: Handle edge case: If we remove all nodes, we should inform the user
    # and then exit
    if not filtered_nodes:
        print("All nodes removed. Exiting.")
        exit(0)

    print(f"Filtered a total of {len(filtered_nodes)} nodes")
    return filtered_nodes
