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
        url: str,
        database: str,
        labels: str = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``."""
        self._database = database
        self._labels = labels
        self._url = url
        self._credentials = deepcopy(credentials) or {}

        super().__init__(
            filepath="filepath",
            save_args=save_args,
            load_args=load_args,
            credentials=credentials,
            version=version,
            metadata=metadata,
        )

    def _load(self) -> Any:
        spark_session = _get_spark()

        return (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .option("labels", self._labels)
            .options(**self._credentials)
            .load(**self._load_args)
        )

    def _save(self, data: DataFrame) -> None:
        (
            data.write.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .option("labels", self._labels)
            .option("node.keys", "id:id")
            .options(**self._credentials)
            .save(**self._save_args)
        )
