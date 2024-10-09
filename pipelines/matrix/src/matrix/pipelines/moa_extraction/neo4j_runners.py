"""Module containing strategies for interacting with Neo4j."""

from neo4j import GraphDatabase

from abc import ABC, abstractmethod


class Neo4jRunner(ABC):
    """Abstract base class for Neo4j runners."""

    @abstractmethod
    def run(self, query: str, *args, **kwargs):
        """Run a query on the Neo4j database.

        Args:
            query: The query to run.
        """
        pass


class LocalNeo4jRunner(Neo4jRunner):
    """Neo4j runner for local Neo4j instances."""

    def __init__(self, uri: str, user: str, password: str, database: str):
        """Initialize the Neo4j runner.

        Args:
            uri: The URI of the Neo4j instance.
            user: The user to connect to the Neo4j instance.
            password: The password to connect to the Neo4j instance.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

    def run(self, query: str):
        """Run a query on the Neo4j database.

        Args:
            query: The query to run.
        """
        with self.driver.session() as session:
            info = session.run(query)
            x = info.values()
        return x
