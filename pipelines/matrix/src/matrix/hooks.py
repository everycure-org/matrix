"""Kedro project hooks."""
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
from kedro.pipeline.node import Node
from datetime import datetime
from typing import Any, Optional, Dict
import pandas as pd
import termplotlib as tpl

from kedro.framework.context import KedroContext
from kedro.io.data_catalog import DataCatalog
from kedro_datasets.spark import SparkDataset


class SparkHooks:
    """Spark project hook with lazy initialization and global session override."""

    _spark_session: Optional[SparkSession] = None

    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialise SparkSession.

        Initialises a SparkSession using the config defined
        in project's conf folder.
        """
        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())
        self.context = context

    @classmethod
    def _initialize_spark(cls, context: KedroContext) -> None:
        """Initialize SparkSession if not already initialized and set as default."""
        # if we have not initiated one, we initiate one
        if cls._spark_session is None:
            print("we are killing spark to create a fresh one")
            # Clear any existing default session, we take control!
            sess = SparkSession.getActiveSession()
            if sess is not None:
                sess.stop()

            parameters = context.config_loader["spark"]
            if context.env == "prod":
                parameters = {
                    k: v
                    for k, v in parameters.items()
                    if not k.startswith("spark.hadoop.google.cloud.auth.service")
                }
            print(f"starting spark session with the following parameters: {parameters}")
            spark_conf = SparkConf().setAll(parameters.items())

            # Create and set our configured session as the default
            cls._spark_session = (
                SparkSession.builder.appName(context.project_path.name)
                .config(conf=spark_conf)
                .getOrCreate()
            )

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
        dataset = self.catalog._get_dataset(dataset_name)
        if isinstance(dataset, SparkDataset):
            print(f"SparkDataset detected: {dataset}")
            self._initialize_spark(self.context)

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
