"""Kedro project hooks."""
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
import os
from pyspark.sql import SparkSession
from kedro.pipeline.node import Node
from datetime import datetime
from typing import Any, Optional, Dict
import pandas as pd
import termplotlib as tpl
from omegaconf import OmegaConf
import logging

import mlflow

from kedro.framework.context import KedroContext
from kedro.io.data_catalog import DataCatalog
from kedro_datasets.spark import SparkDataset


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

    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialise MLFlow run.

        Initialises a MLFlow run and passes it on for
        other hooks to consume.
        """
        cfg = OmegaConf.create(context.config_loader["mlflow"])
        globs = OmegaConf.create(context.config_loader["globals"])

        # Set tracking uri
        # NOTE: This piece of code ensures that every MLFlow experiment
        # is created by our Kedro pipeline with the right artifact root.
        mlflow.set_tracking_uri(cfg.server.mlflow_tracking_uri)
        experiment_id = self._create_experiment(
            cfg.tracking.experiment.name, globs.mlflow_artifact_root
        )

        if cfg.tracking.run.name:
            run_id = self._create_run(cfg.tracking.run.name, experiment_id)

            # Update catalog
            OmegaConf.update(cfg, "tracking.run.id", run_id)
            context.config_loader["mlflow"] = cfg

    @staticmethod
    def _create_run(run_name: str, experiment_id: str) -> str:
        """Function to create run for given run_name.

        Args:
            run_name: name of the run
            experiment_id: id of the experiment
        Returns:
            Identifier of created run
        """
        # Retrieve run
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"run_name='{run_name}'",
            order_by=["start_time DESC"],
            output_format="list",
        )

        if not runs:
            with mlflow.start_run(
                run_name=run_name, experiment_id=experiment_id
            ) as run:
                mlflow.set_tag("created_by", "kedro")
                return run.info.run_id

        return runs[0].info.run_id

    @staticmethod
    def _create_experiment(experiment_name: str, artifact_location: str) -> str:
        """Function to create experiment.

        Args:
            experiment_name: name of the experiment
            artifact_location: artifact location of experiment
        Returns:
            Identifier of experiment
        """
        experiments = mlflow.search_experiments(
            filter_string=f"name = '{experiment_name}'"
        )

        if not experiments:
            return mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )

        return experiments[0].experiment_id


class SparkHooks:
    """Spark project hook with lazy initialization and global session override."""

    _spark_session: Optional[SparkSession] = None
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
            sess = SparkSession.getActiveSession()
            if sess is not None:
                logger.warning("we are killing spark to create a fresh one")
                sess.stop()
            parameters = cls._kedro_context.config_loader["spark"]

            # DEBT ugly fix, ideally we overwrite this in the spark.yml config file but currently no
            # known way of doing so
            # if prod environment, remove all config keys that start with spark.hadoop.google.cloud.auth.service
            if (
                cls._kedro_context.env == "cloud"
                and os.environ.get("ARGO_NODE_ID") is not None
            ):
                logger.warning(
                    "we're manipulating the spark configuration now. this is done assuming this is a production execution in argo"
                )
                parameters = {
                    k: v
                    for k, v in parameters.items()
                    if not k.startswith("spark.hadoop.google.cloud.auth.service")
                }
            else:
                logger.info(f"Executing for enviornment: {cls._kedro_context.env}")
                logger.info(
                    f'With ARGO_POD_UID set to: {os.environ.get("ARGO_NODE_ID", "")}'
                )
                logger.info(
                    "Thus determined not to be in k8s cluster and executing with service-account.json file"
                )

            logging.info(
                f"starting spark session with the following parameters: {parameters}"
            )
            spark_conf = SparkConf().setAll(parameters.items())

            # Create and set our configured session as the default
            cls._spark_session = (
                SparkSession.builder.appName(cls._kedro_context.project_path.name)
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
        if self.__class__._already_initialized is True:
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
        print(f"==========================================")

        df = (
            pd.DataFrame.from_dict(durations, orient="index")
            .rename(columns={0: "time"})
            .sort_values("time")
        )
        df["in_seconds"] = df["time"].apply(lambda x: x.total_seconds())
        df["perc_of_total"] = df["in_seconds"] / df["in_seconds"].sum() * 100
        print(f"Node durations:")
        print(df)
        print(f"==========================================")
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
