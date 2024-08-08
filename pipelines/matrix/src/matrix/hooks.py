"""Kedro project hooks."""
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
from kedro.pipeline.node import Node
from datetime import datetime
from typing import Any
import pandas as pd
import termplotlib as tpl
from omegaconf import OmegaConf

import mlflow


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

        if cfg.tracking.run.name:
            # Set tracking uri
            mlflow.set_tracking_uri(cfg.server.mlflow_tracking_uri)
            experiment_id = self._create_experiment(cfg.tracking.experiment.name)
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
    def _create_experiment(experiment_name: str) -> str:
        """Function to create experiment.

        Args:
            experiment_name: name of the experiment
        Returns:
            Identifier of experiment
        """
        experiments = mlflow.search_experiments(
            filter_string=f"name = '{experiment_name}'"
        )

        if not experiments:
            return mlflow.create_experiment(experiment_name)

        return experiments[0].experiment_id


from kedro.framework.context import KedroContext


class SparkHooks:
    """Spark project hook."""

    @hook_impl
    def after_context_created(self, context: KedroContext) -> None:
        """Initialise SparkSession.

        Initialises a SparkSession using the config defined
        in project's conf folder.
        """
        # Load the spark configuration in spark.yaml using the config loader

        parameters = context.config_loader["spark"]
        # DEBT ugly fix, ideally we overwrite this in the spark.yml config file but currently no
        # known way of doing so
        # if prod environment, remove all config keys that start with spark.hadoop.google.cloud.auth.service
        if context.env == "prod":
            parameters = {
                k: v
                for k, v in parameters.items()
                if not k.startswith("spark.hadoop.google.cloud.auth.service")
            }
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = SparkSession.builder.appName(
            context.project_path.name
        ).config(conf=spark_conf)
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")


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
