"""Utility functions for the MOA extraction pipeline."""

from typing import Any, List

from graphdatascience import QueryRunner
from neo4j import Driver, GraphDatabase

from pyspark.sql import functions as F


class GraphDB:
    """Adaptor class to allow injecting the GraphDB object.

    This is due to a drawback where refit cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str | Driver | QueryRunner,
        auth: F.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDB` instance."""
        self._endpoint = endpoint
        self._auth = tuple(auth)
        self._database = database
        self.driver = GraphDatabase.driver(self._endpoint, auth=self._auth, database=self._database)

    def run(self, query: str) -> List[Any]:
        """Run a query on the Neo4j database and get data.

        Args:
            query: The query to run.
        """
        with self.driver.session() as session:
            return session.run(query).data()
