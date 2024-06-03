"""Module containing Neo4JDataset."""
from typing import Any, Dict, Callable
from copy import deepcopy

from pyspark.sql import DataFrame

from kedro_datasets.spark import SparkDataset
from kedro.io.core import Version
from kedro_datasets.spark.spark_dataset import _get_spark

from pypher import Pypher

from refit.v1.core.inject import inject_object


class Neo4JSparkDataset(SparkDataset):
    """Dataset to load and save data from Neo4J.

    Kedro dataset to load and write data from Neo4J. This is essentially a wrapper
    for the [Neo4J spark connector](https://neo4j.com/docs/spark/current/).

    FUTURE:
        - Allow defining a [schema](https://neo4j.com/docs/spark/current/read/define-schema/)
        - Test [predicate pushdown](https://neo4j.com/docs/spark/current/performance/spark/)
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(  # noqa: PLR0913
        self,
        *,
        url: str,
        database: str,
        query: Dict[str, str] = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        version: Version = None,
        credentials: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``Neo4JDataset``.

        Example reading from a query:
        ::
            query:
                object: path.to.pypher.function
        ::

        Example writing nodes:
        ::
            metadata:
                labels: ":Disease"
                node.keys: id


        Example writing a relationship:
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

        Args:
            url: URL of the Neo4J instance.
            database: Name of the Neo4J database.
            query: path to function that yields a Pypher query for execution
            labels: Labels to filter the nodes.
            load_args: Arguments to pass to the load method.
            save_args: Arguments to pass to the save
            version: Version of the dataset.
            credentials: Credentials to connect to the Neo4J instance.
            metadata: Metadata to pass to neo4j connector.
        """
        self._database = database
        self._url = url
        self._query = query
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

        read_object = (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .options(**self._credentials)
        )

        if self._query:
            read_object.option("query", str(self._load_query(self._query)))
        else:
            read_object.option(**self.metadata)

        return read_object.load(**self._load_args)

    def _save(self, data: DataFrame) -> None:
        (
            data.write.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .options(**self._credentials)
            .options(**self.metadata)
            .save(**self._save_args)
        )

    @staticmethod
    @inject_object()
    def _load_query(func: Callable) -> Pypher:
        return func(Pypher())
