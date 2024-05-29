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
        labels: str = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``."""
        # Handle default load and save arguments
        self._labels = labels
        super().__init__(filepath="filepath")

    def _load(self) -> Any:
        spark_session = _get_spark()

        return (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", "everycure")
            .option("url", "bolt://127.0.0.1:7687")
            .option("authentication.type", "basic")
            .option("labels", self._labels)
            .option("authentication.basic.username", "neo4j")
            .option("authentication.basic.password", "admin")
            .load(self._load_args)
        )

    def _save(self, data: DataFrame) -> None:
        (
            data.write.mode("overwrite")
            .format("org.neo4j.spark.DataSource")
            .option("database", "everycure")
            .option("url", "bolt://127.0.0.1:7687")
            .option("authentication.type", "basic")
            .option("labels", self._labels)
            .option("authentication.basic.username", "neo4j")
            .option("authentication.basic.password", "admin")
            .option("node.keys", "id:id")
            .save()
        )
