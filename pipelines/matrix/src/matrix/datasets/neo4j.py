"""Module containing Neo4JDataset."""
from typing import Any, Callable, Union, Optional
from copy import deepcopy
from functools import wraps

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from kedro.io.core import Version
from kedro_datasets.spark import SparkDataset


class Neo4JSparkDataset(SparkDataset):
    """Dataset to load and save data from Neo4J.

    Kedro dataset to load and write data from Neo4J. This is essentially a wrapper
    for the [Neo4J spark connector](https://neo4j.com/docs/spark/current/).
    """

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

        Example reading data:
        ::

            # The node function
            @cypher_query(query="MATCH p=()-[r:TREATS]->() RETURN p")
            def node(data: Dataframe):
                ... # Node logic here
        ::

        Example reading data with parameters:
        ::

            def query(node_label: str, **kwargs):
                return f"MATCH p=(n)-[r:TREATS]->() WHERE n.category in ['{node_label}'] RETURN p"

            # The node function
            @cypher_query(query=query)
            def node(data: Dataframe, node_label: str):
                ... # Node logic here
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
        spark_session = SparkSession.builder.getOrCreate()

        return (
            spark_session.read.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .options(**self._credentials)
            .options(**self._load_args)
            .load()
        )

    def _save(self, data: DataFrame) -> None:
        (
            data.write.format("org.neo4j.spark.DataSource")
            .option("database", self._database)
            .option("url", self._url)
            .options(**self._credentials)
            .options(**self._save_args)
            .save(**{"mode": "overwrite"})
        )


# def cypher_query(query: Union[str, Callable], schema: Optional[StructType] = None):
#     """Decorator to specify Cypher query for Neo4J dataset.

#     The Cypher query annotator is required to use for nodes that read datasets of
#     the Neo4JSparkDataset type. The `query` argument can either be a string representing
#     a Cypher query, or a callable that yields the Cypher query. The callable will be passed
#     all of the arguments passed into the node, such that they can be used to interpolate the query.

#     Args:
#         query: Cypher query to use.
#         schema: Optional PySpark [schema](https://neo4j.com/docs/spark/current/read/define-schema/) to convert to.
#     """

#     def decorate(func):
#         @wraps(func)
#         def wrapper(read_obj, *args, **kwargs):
#             if callable(query):
#                 read_obj.option("query", query(*args, **kwargs))
#             else:
#                 read_obj.option("query", query)

#             if schema:
#                 read_obj.schema(schema)

#             return func(read_obj.load(), *args, **kwargs)

#         return wrapper

#     return decorate
