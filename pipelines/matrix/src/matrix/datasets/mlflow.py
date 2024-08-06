"""Custom Mlflow datasets."""
import mlflow
from copy import deepcopy
from typing import Any, Dict, Union

from mlflow.tracking import MlflowClient

from kedro_mlflow.io.metrics.mlflow_abstract_metric_dataset import (
    MlflowAbstractMetricDataset,
)


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
        pass
        # if self._logging_activated:
        #     self._validate_run_id()
        #     run_id = self.run_id  # we access it once instead of calling self.run_id everywhere to avoid looking or an active run each time

        #     mlflow_client = MlflowClient()

        #     save_args = deepcopy(self._save_args)
        #     if not self.mode == "overwrite":
        #         raise NotImplementedError()

        #     for key, value in data.items():
        #         mlflow_client.log_metric(
        #             run_id=run_id,
        #             key=f"{self._key_prefix}_{key}",
        #             value=value,
        #             step=None,
        #             **save_args,
        #         )
