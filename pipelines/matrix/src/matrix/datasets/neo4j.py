"""Module containing Neo4JDataset."""
from typing import Any
from copy import deepcopy

from pyspark.sql import DataFrame

from kedro_datasets.spark import SparkDataset
from kedro.io.core import Version
from kedro_datasets.spark.spark_dataset import _get_spark


class Neo4JDatabase(SparkDataset):
    """Class to represent Neo4j database."""

    def __init__(self, url: str, database: str) -> None:
        """Creates a new instance of ``Neo4JDatabase``."""
        pass

    def query(self, query: str) -> DataFrame:
        """Execute cypher query on the database."""
        pass

    def to_dataframe(self) -> DataFrame:
        """Convert to dataframe."""
        pass


class Neo4JDataset(SparkDataset):
    """Dataset to load and save data from Neo4J."""

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(  # noqa: PLR0913
        self,
        *,
        url: str,
        database: str,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``.

        Example:
        ::
            metadata:
                relationship: TREATS
                relationship.save.strategy: keys
                relationship.source.save.mode: overwrite
                relationship.source.labels: ":Drug"
                relationship.source.node.keys: source_id:id
                relationship.target.save.mode: overwrite
                relationship.target.labels: ":Disease"
                relationship.target.node.keys: target_id:id
                relationship.nodes.map: true
        ::

        Example:
        ::
            metadata:
                partitions: 4
                query: "MATCH (n:Drug) RETURN n"
        ::

        Args:
            url: URL of the Neo4J instance.
            database: Name of the Neo4J database.
            labels: Labels to filter the nodes.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
        """
        self._database = database
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
            .options(**self.metadata)
            .options(**self._credentials)
            .load(**self._load_args)
        )

    def _save(self, data: DataFrame) -> None:
        (
            data.write.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .options(**self._credentials)
            .options(**self.metadata)
            .save(**self._save_args)
        )
