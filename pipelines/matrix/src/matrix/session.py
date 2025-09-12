from typing import Any, Iterable

from kedro import __version__ as kedro_version
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.session.session import KedroSessionError
from kedro.io import DataCatalog
from kedro.runner import AbstractRunner, SequentialRunner


class KedroSessionWithFromCatalog(KedroSession):
    """Custom Kedro Session.

    Custom Kedro Session that allows an additional `from-catalog` to
    be specified. The from catalog overrides the catalog entry for all input
    datasets.

    NOTE: This module has some code duplication due to Kedros' complex
    config setup. We should cleanup based on:

    https://github.com/kedro-org/kedro/issues/4155
    """

    def run(  # noqa: PLR0913
        self,
        from_catalog: DataCatalog,
        pipeline_name: str | None = None,
        tags: Iterable[str] | None = None,
        runner: AbstractRunner | None = None,
        node_names: Iterable[str] | None = None,
        from_nodes: Iterable[str] | None = None,
        to_nodes: Iterable[str] | None = None,
        from_inputs: Iterable[str] | None = None,
        to_outputs: Iterable[str] | None = None,
        load_versions: dict[str, str] | None = None,
        namespace: str | None = None,
        from_run_datasets: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Runs the pipeline with a specified runner.

        Args:
            from_catalog: From catalog to use, if set will override input datasets.
            from_params: From params to set, will override params.
            pipeline_name: Name of the pipeline that is being run.
            tags: An optional list of node tags which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                containing *any* of these tags will be run.
            runner: An optional parameter specifying the runner that you want to run
                the pipeline with.
            node_names: An optional list of node names which should be used to
                filter the nodes of the ``Pipeline``. If specified, only the nodes
                with these names will be run.
            from_nodes: An optional list of node names which should be used as a
                starting point of the new ``Pipeline``.
            to_nodes: An optional list of node names which should be used as an
                end point of the new ``Pipeline``.
            from_inputs: An optional list of input datasets which should be
                used as a starting point of the new ``Pipeline``.
            to_outputs: An optional list of output datasets which should be
                used as an end point of the new ``Pipeline``.
            load_versions: An optional flag to specify a particular dataset
                version timestamp to load.
            namespace: The namespace of the nodes that is being run.

        Raises:
            ValueError: If the named or `__default__` pipeline is not
                defined by `register_pipelines`.
            Exception: Any uncaught exception during the run will be re-raised
                after being passed to ``on_pipeline_error`` hook.
            KedroSessionError: If more than one run is attempted to be executed during
                a single session.

        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.
        """
        # Report project name
        self._logger.info("Kedro project %s", self._project_path.name)

        if self._run_called:
            raise KedroSessionError(
                "A run has already been completed as part of the"
                " active KedroSession. KedroSession has a 1-1 mapping with"
                " runs, and thus only one run should be executed per session."
            )

        session_id = self.store["session_id"]
        save_version = session_id
        extra_params = self.store.get("extra_params") or {}
        extra_params["pipeline_name"] = pipeline_name
        self._store["extra_params"] = extra_params
        context = self.load_context()
        name = pipeline_name or "__default__"

        try:
            pipeline = pipelines[name]
        except KeyError as exc:
            raise ValueError(
                f"Failed to find the pipeline named '{name}'. "
                f"It needs to be generated and returned "
                f"by the 'register_pipelines' function."
            ) from exc

        filtered_pipeline = pipeline.filter(
            tags=tags,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            node_names=node_names,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            node_namespace=namespace,
        )

        record_data = {
            "session_id": session_id,
            "project_path": self._project_path.as_posix(),
            "env": context.env,
            "kedro_version": kedro_version,
            "tags": tags,
            "from_nodes": from_nodes,
            "to_nodes": to_nodes,
            "node_names": node_names,
            "from_inputs": from_inputs,
            "to_outputs": to_outputs,
            "load_versions": load_versions,
            "extra_params": extra_params,
            "pipeline_name": pipeline_name,
            "namespace": namespace,
            "runner": getattr(runner, "__name__", str(runner)),
        }

        catalog = context._get_catalog(
            save_version=save_version,
            load_versions=load_versions,
        )

        if from_catalog:
            if from_run_datasets:
                for item in from_run_datasets:
                    # Update specified datasets to read from tge from_catalog
                    # Applicable on cluster runs where each node is it's own pipeline and we need to determine the relevant input datasets upfront
                    self._logger.info("Replacing %s", item)
                    catalog.add(item, from_catalog._get_dataset(item), replace=True)
            else:
                # Update all pipeline inputs to read from
                # the from catalog
                for item in filtered_pipeline.inputs():
                    self._logger.info("Replacing %s", item)
                    catalog.add(item, from_catalog._get_dataset(item), replace=True)

        # Run the runner
        hook_manager = self._hook_manager
        runner = runner or SequentialRunner()
        if not isinstance(runner, AbstractRunner):
            raise KedroSessionError(
                "KedroSession expect an instance of Runner instead of a class."
                "Have you forgotten the `()` at the end of the statement?"
            )
        hook_manager.hook.before_pipeline_run(run_params=record_data, pipeline=filtered_pipeline, catalog=catalog)

        try:
            run_result = runner.run(filtered_pipeline, catalog, hook_manager, session_id)
            self._run_called = True
        except Exception as error:
            hook_manager.hook.on_pipeline_error(
                error=error,
                run_params=record_data,
                pipeline=filtered_pipeline,
                catalog=catalog,
            )
            raise

        hook_manager.hook.after_pipeline_run(
            run_params=record_data,
            run_result=run_result,
            pipeline=filtered_pipeline,
            catalog=catalog,
        )
        return run_result
