import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Collection, Dict, Optional, Set

import fsspec
import mlflow
import pandas as pd
import pyspark.sql as ps
import termplotlib as tpl
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.framework.project import pipelines
from kedro.io.data_catalog import DataCatalog
from kedro.pipeline.node import Node
from kedro_datasets.spark import SparkDataset
from omegaconf import OmegaConf
from pyspark import SparkConf

from matrix.inject import _parse_for_objects
from matrix.pipelines.data_release import last_node_name as last_data_release_node_name

logger = logging.getLogger(__name__)


class MLFlowHooks:
    """Kedro MLFlow hook.

    Mlflow supports the concept of run names, which are mapped
    to identifiers behind the curtains. However, this name is not
    required to be unique and hence multiple runs for the same name
    may exist. This plugin ensures run names are mapped to single
    a identifier. Can be removed when following issue is resolved:

    https://github.com/Galileo-Galilei/kedro-mlflow/issues/579
    """

    _kedro_context: Optional[KedroContext] = None
    _input_datasets: Optional[Set] = None

    @classmethod
    def set_context(cls, context: KedroContext) -> None:
        """Utility class method that stores context in class as singleton."""
        cls._kedro_context = context

    @classmethod
    def get_pipeline_inputs(cls):
        if cls._input_datasets is None:
            pipeline_name = cls._kedro_context._extra_params["pipeline_name"]
            pipeline_obj = pipelines[pipeline_name]
            inputs = {input for input in pipeline_obj.all_inputs() if not input.startswith("params:")}
            outputs = pipeline_obj.all_outputs()
            inputs_only = inputs - outputs
            cls._input_datasets = inputs_only

    # @hook_impl disabled due to https://github.com/everycure-org/matrix/issues/1154
    def after_dataset_loaded(self, dataset_name, data, node):
        """In the v1 of this feature we only log the dataset names.
        Logging actual datasets requires extra work, due to the fact that the data has to be first
        converted to mlflow.data.dataset.Dataset class. This works for common formats like pandas and spark
        where the from_ functions exist, e.g.:

        dataset = mlflow.data.from_pandas(data, name=dataset_name)
        mlflow.log_input(dataset)

        but our datasets are too heterogenous and can't be always parsed, e.g. matrix.datasets.graph.KnowledgeGraph
        or would need a lot of hard-coded logic and would make this code brittle.
        One idea is to only log select certain dataset types - pandas and spark, for other keep logging names only.

        Another improvement idea for v2 would be to additionally  log the dataset paths, e.g.:
        path = self._kedro_context.catalog.datasets[dataset_name]._get_load_path()
        or using the url for remote datasets:
        path = self._kedro_context.catalog.datasets[dataset_name]._url
        """
        if dataset_name in MLFlowHooks._input_datasets:
            logger.info(f"Processing dataset {dataset_name}")
            if dataset_name not in self.fetch_logged_datasets():
                logger.info(f"Dataset {dataset_name} is not in the already logged datasets. Logging it now:")
                dataset = mlflow.data.from_pandas(pd.DataFrame(), name=dataset_name)
                try:
                    mlflow.log_input(dataset)
                except Exception as ex:
                    logger.error(f"Error encountered when logging dataset {dataset_name}: {ex}")
                    raise
            else:
                logger.info(f"Dataset {dataset_name} has already been logged as input.")

    @staticmethod
    def fetch_logged_datasets() -> set[str]:
        run_id = MLFlowHooks._kedro_context.mlflow.tracking.run.id
        client = mlflow.tracking.MlflowClient()
        logged_inputs = client.get_run(run_id).inputs
        logged_names = {dataset.dataset.name for dataset in logged_inputs.dataset_inputs}
        logger.debug(f"These are dataset names that have already been logged for run id '{run_id}': {logged_names}")
        return logged_names

    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialise MLFlow run.

        Initialises a MLFlow run and passes it on for
        other hooks to consume.
        """
        MLFlowHooks.set_context(context)
        cfg = OmegaConf.create(context.config_loader["mlflow"])
        mlflow.set_tracking_uri(cfg.server.mlflow_tracking_uri)
        mlflow.start_run(run_id=cfg.tracking.run.id, nested=True)


class SparkHooks:
    """Spark project hook with lazy initialization and global session override."""

    _spark_session: Optional[ps.SparkSession] = None
    _already_initialized = False
    _kedro_context: Optional[KedroContext] = None

    @classmethod
    def set_context(cls, context: KedroContext) -> None:
        """Utility class method that stores context in class as singleton."""
        cls._kedro_context = context

    @classmethod
    def _initialize_spark(cls) -> None:
        """Initialize SparkSession if not already initialized and set as default."""
        if cls._kedro_context is None:
            raise Exception("kedro context needs to be set before")
        # if we have not initiated one, we initiate one
        if cls._spark_session is None:
            # Clear any existing default session, we take control!
            sess = ps.SparkSession.getActiveSession()
            if sess is not None:
                # Kedro integration tests (those using
                # kedro.framework.session.KedroSession) that make use of this
                # hook will make all tests using the SparkSession fail if the
                # session started by the pytest fixture is stopped. We may
                # consider to modify the configuration of the SparkSession fixture
                # using the parameters from spark.yml (as is done below), but
                # for the moment this is not needed in our test suite.
                # In other words, referring to the previous comment: we do not
                # take control in the case of a test suite.
                if "PYTEST_CURRENT_TEST" in os.environ:
                    cls._spark_session = sess
                    return
                logger.warning("we are killing spark to create a fresh one")
                sess.stop()
            parameters = cls._kedro_context.config_loader["spark"]

            # DEBT ugly fix, ideally we overwrite this in the spark.yml config file but currently no
            # known way of doing so
            # if prod environment, remove all config keys that start with spark.hadoop.google.cloud.auth.service
            if "ARGO_NODE_ID" in os.environ:
                logger.warning(
                    "We're manipulating the spark configuration now. This is done assuming this is a production execution in argo"
                )
                parameters = {
                    k: v for k, v in parameters.items() if not k.startswith("spark.hadoop.google.cloud.auth.service")
                }
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "conf/local/service-account.json"
            else:
                logger.info(f"Executing for environment: {cls._kedro_context.env}")
                logger.info(f'With ARGO_POD_UID set to: {os.environ.get("ARGO_NODE_ID", "")}')
                logger.info("Thus determined not to be in k8s cluster and executing with service-account.json file")

            # When running `kedro run`, the SparkSession is created with the impersonation service account.
            if os.environ.get("SPARK_IMPERSONATION_SERVICE_ACCOUNT") is not None:
                service_account = os.environ["SPARK_IMPERSONATION_SERVICE_ACCOUNT"]
                parameters["spark.hadoop.fs.gs.auth.impersonation.service.account"] = service_account
                logger.info(f"Using service account: {service_account} for spark impersonation")

            logger.info(f"starting spark session with the following parameters: {parameters}")
            spark_conf = SparkConf().setAll(parameters.items())

            # Create and set our configured session as the default
            cls._spark_session = (
                ps.SparkSession.builder.appName(cls._kedro_context.project_path.name)
                .config(conf=spark_conf)
                .getOrCreate()
            )
        else:
            logger.debug("SparkSession already initialized")

    @hook_impl
    def after_context_created(self, context: KedroContext) -> None:
        """Remember context for later spark initialization."""
        logger.info("Remembering context for spark later")
        SparkHooks.set_context(context)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        feed_dict: Dict[str, Any],
        save_version: str,
        load_versions: Dict[str, str],
    ) -> None:
        """Store the KedroContext for later use."""
        self.catalog = catalog

    def _check_and_initialize_spark(self, dataset_name: str) -> None:
        """Check if dataset is SparkDataset and initialize Spark if needed."""
        if self.__class__._already_initialized:
            return

        dataset = self.catalog._get_dataset(dataset_name)
        if isinstance(dataset, SparkDataset):
            logger.info(f"SparkDataset detected: {dataset}")
            self._initialize_spark()
            self.__class__._already_initialized = True

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node: Any) -> None:
        """Initialize Spark if the dataset is a SparkDataset."""
        self._check_and_initialize_spark(dataset_name)

    @hook_impl
    def before_dataset_saved(self, dataset_name: str, data: Any, node: Any) -> None:
        """Initialize Spark if the dataset is a SparkDataset."""
        self._check_and_initialize_spark(dataset_name)

    @hook_impl
    def after_dataset_loaded(self):
        """Print the current Spark configuration after each dataset is loaded.
        Must be done after dataset is loaded, because we initialise spark only
        for Spark dataset types."""
        try:
            msg = ["Current Spark Configuration:"]
            msg.extend([f"{k}: {v}" for k, v in sorted(self._spark_session.sparkContext.getConf().getAll())])
            logger.debug("\n".join(msg))
        except AttributeError:
            logger.warning("SparkSession is not initialized.")


class NodeTimerHooks:
    """Spark project hook."""

    node_times = {}
    pipeline_start = datetime.now()
    pipeline_end = None

    @hook_impl
    def after_pipeline_run(self, *args) -> None:
        """Calculates the duration per node and shows it."""
        self.pipeline_end = datetime.now()
        durations = {}
        for node in self.node_times:
            nt = self.node_times[node]
            durations[node] = nt["end"] - nt["start"]

        print(f"Total pipeline duration: {self.pipeline_end - self.pipeline_start}")
        print("==========================================")

        df = pd.DataFrame.from_dict(durations, orient="index").rename(columns={0: "time"}).sort_values("time")
        df["in_seconds"] = df["time"].apply(lambda x: x.total_seconds())
        df["perc_of_total"] = df["in_seconds"] / df["in_seconds"].sum() * 100
        print("Node durations:")
        print(df)
        print("==========================================")
        fig = tpl.figure()
        fig.barh(df["in_seconds"], df.index, force_ascii=True)
        fig.show()

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node: Node) -> None:
        """For the first dataset loaded for this node, remember the starting point."""
        self._start(node.name)

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        """For nodes without inputs, we remember the start here."""
        self._start(node.name)

    @hook_impl
    def after_node_run(self, node: Node, *args) -> None:
        """For nodes without inputs, we remember the start here."""
        self._ending(node.name)

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None:
        """For each dataset saved, replace the 2nd item for the node, the last one gets remembered."""
        self._ending(node.name)

    def _start(self, name):
        if name not in self.node_times:
            self.node_times[name] = {"start": datetime.now()}

    def _ending(self, name):
        if name in self.node_times:
            self.node_times[name]["end"] = datetime.now()
        else:
            raise Exception("there should be a node starting timer")


class ReleaseInfoHooks:
    _kedro_context: Optional[KedroContext] = None
    _globals: Optional[dict] = None
    _params: Optional[dict] = None

    @classmethod
    def set_context(cls, context: KedroContext) -> None:
        """Utility class method that stores context in class as singleton."""
        cls._kedro_context = context
        cls._globals = context.config_loader["globals"]
        cls._params = context.config_loader["parameters"]

    @hook_impl
    def after_context_created(self, context: KedroContext) -> None:
        """Remember context for later export from a node hook."""
        logger.info("Remembering context for context export later")
        ReleaseInfoHooks.set_context(context)

    @staticmethod
    def build_bigquery_link() -> str:
        version = ReleaseInfoHooks._globals["versions"]["release"]
        version_formatted = "release_" + re.sub(r"[.-]", "_", version)
        tmpl = (
            f"https://console.cloud.google.com/bigquery?"
            f"project={ReleaseInfoHooks._globals['runtime_gcp_project']}"
            f"&ws=!1m4!1m3!3m2!1s"
            f"mtrx-hub-dev-3of!2s"
            f"{version_formatted}"
        )
        return tmpl

    @staticmethod
    def build_code_link() -> str:
        version = ReleaseInfoHooks._globals["versions"]["release"]
        tmpl = f"https://github.com/everycure-org/matrix/tree/{version}"
        return tmpl

    @staticmethod
    def build_kg_dashboard_link() -> str:
        version = ReleaseInfoHooks._globals["versions"]["release"]
        tmpl = f"https://data.dev.everycure.org/versions/{version}/evidence/"
        return tmpl

    @staticmethod
    def build_mlflow_link() -> str:
        run_id = ReleaseInfoHooks._kedro_context.mlflow.tracking.run.id
        experiment_name = ReleaseInfoHooks._kedro_context.mlflow.tracking.experiment.name
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        tmpl = f"https://mlflow.platform.dev.everycure.org/#/experiments/{experiment_id}/runs/{run_id}"
        return tmpl

    @classmethod
    def extract_used_dataset_names(cls) -> list:
        # Using lazy import to prevent circular import error
        from matrix.settings import DYNAMIC_PIPELINES_MAPPING

        dataset_names = [item["name"] for item in DYNAMIC_PIPELINES_MAPPING()["integration"] if item["integrate_in_kg"]]
        return dataset_names

    @classmethod
    def extract_all_global_datasets(cls, hidden_datasets: Collection) -> dict:
        datasources_to_versions = {
            k: v["version"] for k, v in ReleaseInfoHooks._globals["data_sources"].items() if k not in hidden_datasets
        }
        return datasources_to_versions

    @staticmethod
    def extract_release_info(global_datasets: dict[str, Any]) -> dict[str, str]:
        normalizer = _parse_for_objects(ReleaseInfoHooks._params["integration"]["normalization"]["normalizer"])
        info = {
            "Release Name": ReleaseInfoHooks._globals["versions"]["release"],
            "Datasets": global_datasets,
            "BigQuery": ReleaseInfoHooks.build_bigquery_link(),
            "KG dashboard": ReleaseInfoHooks.build_kg_dashboard_link(),
            "MLFlow": ReleaseInfoHooks.build_mlflow_link(),
            "Code": ReleaseInfoHooks.build_code_link(),
            "NodeNorm Endpoint": f"{normalizer.endpoint} ({normalizer.version()})",
        }
        return info

    @staticmethod
    def upload_to_storage(release_info: dict[str, str]) -> None:
        release_dir = ReleaseInfoHooks._globals["release_dir"]
        release_version = ReleaseInfoHooks._globals["versions"]["release"]
        full_blob_path = os.path.join(release_dir, f"{release_version}_info.json")

        with fsspec.open(full_blob_path, "wb") as f:
            f.write(json.dumps(release_info).encode("utf-8"))

    @hook_impl
    def after_node_run(self, node: Node) -> None:
        """Runs after the last node of the data_release pipeline"""
        # We chose to add this using the `after_node_run` hook, rather than
        # `after_pipeline_run`, because one does not know a priori which
        # pipelines the (last) data release node is part of. With an
        # `after_node_run`, you can limit your filters easily.
        if node.name == last_data_release_node_name:
            datasets_to_hide = frozenset([])
            global_datasets = self.extract_all_global_datasets(datasets_to_hide)
            used_dataset_names = self.extract_used_dataset_names()
            used_datasets = {k: v for k, v in global_datasets.items() if k in used_dataset_names}
            release_info = self.extract_release_info(used_datasets)
            try:
                self.upload_to_storage(release_info)
            except KeyError:
                logger.warning("Could not upload release info after running Kedro node.", exc_info=True)
