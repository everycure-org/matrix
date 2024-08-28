"""Custom Mlflow datasets."""
import mlflow

import pandas as pd
from copy import deepcopy
from typing import Any, Dict, Union

from mlflow.tracking import MlflowClient

from kedro_mlflow.io.metrics.mlflow_abstract_metric_dataset import (
    MlflowAbstractMetricDataset,
)

from kedro_datasets.pandas import ParquetDataset, CSVDataset
from kedro_datasets.spark import SparkDataset
from kedro_datasets.spark.spark_dataset import _strip_dbfs_prefix

from kedro.io.core import PROTOCOL_DELIMITER, AbstractDataset

from refit.v1.core.inject import _parse_for_objects


class MlFlowInputDataDataSet(AbstractDataset):
    """Kedro dataset to represent MLFlow Input Dataset."""

    def __init__(
        self,
        *,
        name: str,
        context: str,
        dataset: AbstractDataset,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialise MlflowMetricDataset.

        Args:
            name (str): name of dataset in MLFlow
            context: context where dataset is used
            dataset: Underlying Kedro dataset
            metadata: kedro metadata
        """
        self._name = name
        self._context = context
        self._dataset = _parse_for_objects(dataset)

    def _load(self) -> Any:
        return self._dataset._load()

    def _save(self, data):
        self._dataset.save(data)

        # FUTURE: Support other datasets
        # FUTURE: Fix the source of data
        # https://github.com/mlflow/mlflow/issues/13015
        if any(isinstance(self._dataset, ds) for ds in [ParquetDataset, CSVDataset]):
            ds = mlflow.data.from_pandas(
                data, name=self._name, source=self._get_full_path(self._dataset)
            )
        elif isinstance(self._dataset, SparkDataset):
            ds = mlflow.data.from_spark(
                data,
                name=self._name,
                path=_strip_dbfs_prefix(
                    self._dataset._fs_prefix + str(self._dataset._get_load_path())
                ),
            )
        else:
            raise NotImplementedError(
                f"MLFlow Logging for dataset of type {type(self._dataset)} not implemented!"
            )

        mlflow.log_input(ds, context=self._context)

    @staticmethod
    def _get_full_path(dataset: AbstractDataset):
        return f"{dataset._protocol}://{str(dataset._filepath)}"

    def _describe(self) -> Dict[str, Any]:
        """Describe MLflow metrics dataset.

        Returns:
            Dict[str, Any]: Dictionary with MLflow metrics dataset description.
        """
        return {"context": self._context, "name": self._name}


class MlflowMetricsDataset(MlflowAbstractMetricDataset):
    """Custom Mlflow dataset to save multiple metrics.

    Current mlflow implementation does not support storing multiple metrics through
    a single dataset. To be decomissioned on addition of the following issue:

    https://github.com/Galileo-Galilei/kedro-mlflow/issues/440
    """

    SUPPORTED_SAVE_MODES = {"overwrite", "append"}
    DEFAULT_SAVE_MODE = "overwrite"

    def __init__(
        self,
        run_id: str = None,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        key_prefix: str = None,
    ):
        """Initialise MlflowMetricDataset.

        Args:
            run_id (str): The ID of the mlflow run where the metric should be logged
            load_args: dataset load arguments
            save_args: dataset save arguments
            key_prefix: key prefix used for all metrics
        """
        self.run_id = run_id
        self._load_args = load_args or {}
        self._save_args = save_args or {}
        self._logging_activated = True
        self._key_prefix = key_prefix

        # We add an extra argument mode="overwrite" / "append" to enable logging update an existing metric
        # this is not an offical mlflow argument for log_metric, so we separate it from the others
        # "overwrite" corresponds to the default mlflow behaviour
        self.mode = self._save_args.pop("mode", self.DEFAULT_SAVE_MODE)

    @property
    def run_id(self) -> Union[str, None]:
        """Get run id."""
        run = mlflow.active_run()
        if (self._run_id is None) and (run is not None):
            # if no run_id is specified, we try to retrieve the current run
            # this is useful because during a kedro run, we want to be able to retrieve
            # the metric from the active run to be able to reload a metric
            # without specifying the (unknown) run id
            return run.info.run_id

        # else we return the _run_id which can eventually be None.
        # In this case, saving will work (a new run will be created)
        # but loading will fail,
        # according to mlflow's behaviour
        return self._run_id

    @run_id.setter
    def run_id(self, run_id: str):
        self._run_id = run_id

    def _load(self):
        raise NotImplementedError()

    def _exists(self) -> bool:
        """Check if the metric exists in remote mlflow storage exists.

        Returns:
            bool: Does the metric name exist in the given run_id?
        """
        return False

    def _describe(self) -> Dict[str, Any]:
        """Describe MLflow metrics dataset.

        Returns:
            Dict[str, Any]: Dictionary with MLflow metrics dataset description.
        """
        return {
            "run_id": self.run_id,
        }

    def _save(self, data: Dict[str, Any]):
        if self._logging_activated:
            self._validate_run_id()
            run_id = self.run_id  # we access it once instead of calling self.run_id everywhere to avoid looking or an active run each time

            mlflow_client = MlflowClient()

            save_args = deepcopy(self._save_args)
            if not self.mode == "overwrite":
                raise NotImplementedError()

            for key, value in data.items():
                mlflow_client.log_metric(
                    run_id=run_id,
                    key=f"{self._key_prefix}_{key}",
                    value=value,
                    step=None,
                    **save_args,
                )
