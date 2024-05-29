"""Module containing Neo4JDataset."""
from typing import Any
from kedro_datasets.spark import SparkDataset
from kedro.io.core import Version

from kedro_datasets.spark.spark_dataset import _get_spark


class Neo4JDataset(SparkDataset):
    """Dataset to load and save data from Neo4J."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        filepath: str,
        file_format: str = "parquet",
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``SparkDataset``."""
        pass

    def _load(self) -> Any:
        spark_session = _get_spark()

        return spark_session.read.format("org.neo4j.spark.DataSource").load()
