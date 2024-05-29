"""Module containing Neo4JDataset."""
from typing import Any
from copy import deepcopy

from pyspark.sql import DataFrame

from kedro_datasets.spark import SparkDataset
from kedro.io.core import Version
from kedro_datasets.spark.spark_dataset import _get_spark


class Neo4JDataset(SparkDataset):
    """Dataset to load and save data from Neo4J."""

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(  # noqa: PLR0913
        self,
        *,
        labels: str,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``SparkDataset``."""
        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)

        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> Any:
        spark_session = _get_spark()

        return spark_session.read.format("org.neo4j.spark.DataSource").load(
            self._load_args
        )

    def _save(self, data: DataFrame) -> None:
        (
            data.write.mode("overwrite")
            .format("org.neo4j.spark.DataSource")
            .option("labels", ":Drug")
            .save(self._save_args)
        )
