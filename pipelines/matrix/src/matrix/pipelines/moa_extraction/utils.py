# """Module with utilities for the moa extraction pipeline."""

# from neo4j import GraphDatabase


# class Neo4jRunner:
#     """Helper class for running neo4j queries."""

#     def __init__(self, uri: str, user: str, password: str, database: str):
#         """Initialize the Neo4j runner.

#         Args:
#             uri: The URI of the Neo4j instance.
#             user: The user to connect to the Neo4j instance.
#             password: The password to connect to the Neo4j instance.
#             database: The name of the database containing the knowledge graph.
#         """
#         self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

#     def run(self, query: str):
#         """Run a query on the Neo4j database.

#         Args:
#             query: The query to run.
#         """
#         with self.driver.session() as session:
#             info = session.run(query)
#             x = info.values()
#         return x
