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

from matrix.session import KedroSessionWithFromCatalog


class RunConfig(NamedTuple):
    env: str
    conf_source: Optional[str]


@project_group.command()
@env_option
@click.option(
    "--conf-source",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help=CONF_SOURCE_HELP,
)
def validate():
    """Run the pipeline."""

    config = RunConfig(
        conf_source=conf_source,
    )
    _validate(config, KedroSessionWithFromCatalog)


def _validate(config: RunConfig, kedro_session: KedroSessionWithFromCatalog) -> None:
    runner = load_obj(config.runner or "SequentialRunner", "kedro.runner")

    with kedro_session.create(env=config.env, conf_source=config.conf_source, extra_params=config.params) as session:
        # introduced to filter out tags that should not be run
        node_names = _filter_nodes_missing_tag(
            without_tags=config.without_tags, pipeline_obj=config.pipeline_obj, node_names=config.node_names
        )

        from_catalog = _extract_config(config, session)

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
        # Load second config loader instance
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
    return from_catalog


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
    nodes_to_remove = set(node.name for node in pipeline_obj.nodes if node.tags.issuperset(without_tags))

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
