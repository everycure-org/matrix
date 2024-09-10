"""Command line tools for manipulating a Kedro project.

Intended to be invoked via `kedro`.
"""
from typing import List, Set
import click
from kedro.framework.cli.project import (
    ASYNC_ARG_HELP,
    CONFIG_FILE_HELP,
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
    CONTEXT_SETTINGS,
    _config_file_callback,
    _split_params,
    _split_load_versions,
    env_option,
    split_string,
    split_node_names,
)
from kedro.utils import load_obj
from kedro.pipeline.pipeline import Pipeline
from kedro.framework.context.context import _convert_paths_to_absolute_posix
from kedro.framework.hooks import _create_hook_manager
from kedro.io.data_catalog import DataCatalog
from kedro.framework.project import (
    pipelines,
    settings,
)

from matrix.context import CustomKedroSession


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@project_group.command()
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes",
    type=str,
    default="",
    help=FROM_NODES_HELP,
    callback=split_node_names,
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_node_names
)
@click.option(
    "--nodes",
    "-n",
    "node_names",
    type=str,
    multiple=False,
    help=NODE_ARG_HELP,
    callback=split_string,
    default="",
)
@click.option(
    "--runner", "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP
)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option("--tags", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option(
    "--without-tags",
    type=str,
    help="used to filter out nodes with tags that should not be run. All dependent downstream nodes are also removed. Note nodes need to have _all_ tags to be removed.",
    callback=split_string,
    default=[],
)
@click.option(
    "--load-versions",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_split_load_versions,
)
@click.option("--pipeline", "-p", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=CONFIG_FILE_HELP,
    callback=_config_file_callback,
)
@click.option(
    "--conf-source",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help=CONF_SOURCE_HELP,
)
@click.option(
    "--params",
    type=click.UNPROCESSED,
    default="",
    help=PARAMS_ARG_HELP,
    callback=_split_params,
)
@click.option(
    "--to-env",
    type=str,
    default=None,
    help="Custom env to write from, if specified will read from the `env` and write to the `--to-env`",
)
def run(
    tags,
    without_tags,
    env,
    runner,
    is_async,
    node_names,
    to_nodes,
    from_nodes,
    from_inputs,
    to_outputs,
    load_versions,
    pipeline,
    config,
    conf_source,
    params,
    to_env,
):
    """Run the pipeline."""
    if pipeline in ["test", "fabricator"] and env in [None, "base"]:
        raise RuntimeError(
            "Running the fabricator in the base environment might overwrite production data! Use the test env `-e test` instead."
        )

    runner = load_obj(runner or "SequentialRunner", "kedro.runner")
    tags = tuple(tags)
    without_tags = without_tags
    node_names = tuple(node_names)

    with CustomKedroSession.create(
        env=env, conf_source=conf_source, extra_params=params
    ) as session:
        # introduced to filter out tags that should not be run
        node_names = _filter_nodes_missing_tag(
            tuple(without_tags), pipeline, session, node_names
        )

        if to_env:
            # Load second config loader instance
            config_loader_class = settings.CONFIG_LOADER_CLASS
            config_loader = config_loader_class(  # type: ignore[no-any-return]
                conf_source=session._conf_source,
                env=to_env,
                **settings.CONFIG_LOADER_ARGS,
            )
            conf_catalog = config_loader["catalog"]
            conf_catalog = _convert_paths_to_absolute_posix(
                project_path=session._project_path, conf_dictionary=conf_catalog
            )
            conf_creds = config_loader["credentials"]
            catalog: DataCatalog = settings.DATA_CATALOG_CLASS.from_config(
                catalog=conf_catalog, credentials=conf_creds
            )

            # Instantiate pipeline to run
            name = pipeline or "__default__"
            pipe = pipelines[name]
            nodes = pipe.filter(
                tags=tags,
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                node_names=node_names,
                from_inputs=from_inputs,
                to_outputs=to_outputs,
            )

            # Substiute catalog entries
            for item in nodes.all_outputs():
                # NOTE: The `--to-env` can never substitute
                # any parameters, as parameters are read from
                # by definition.
                if item.startswith("params:"):
                    continue

                session._context._catalog.add(
                    item, catalog._get_dataset(item), replace=True
                )

            breakpoint()

        session.run(
            tags=tags,
            runner=runner(is_async=is_async),
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            load_versions=load_versions,
            pipeline_name=pipeline,
        )


def _filter_nodes_missing_tag(
    without_tags: List[str], pipeline: str, session, node_names: List[str]
) -> List[str]:
    """Filter out nodes that have tags that should not be run and their downstream nodes."""
    if not without_tags:
        return node_names

    without_tags: Set[str] = set(without_tags)
    ctx = session.load_context()

    pipeline_name = pipeline or "__default__"
    pipeline_obj: Pipeline = pipelines[pipeline_name]

    if len(node_names) == 0:
        node_names = [node.name for node in pipeline_obj.nodes]

    def should_keep_node(node):
        """Remove node if node has all without_tags tags."""
        if without_tags.intersection(node.tags) == without_tags:
            return False
        else:
            return True

    # Step 1: Identify nodes to remove
    nodes_to_remove = set(
        node.name for node in pipeline_obj.nodes if not should_keep_node(node)
    )

    # Step 2: Identify and add downstream nodes
    downstream_nodes = set()
    downstream_nodes = pipeline_obj.from_nodes(*list(nodes_to_remove)).nodes
    ds_nodes_names = [node.name for node in downstream_nodes]

    nodes_to_remove.update(ds_nodes_names)

    # Step 3: Filter the node_names
    filtered_nodes = [node for node in node_names if node not in nodes_to_remove]

    # Step 4: Handle edge case: If we remove all nodes, we should inform the user
    # and then exit
    if len(filtered_nodes) == 0:
        print("All nodes removed. Exiting.")
        exit(0)

    print(f"Filtered a total of {len(filtered_nodes)} nodes")
    return filtered_nodes
