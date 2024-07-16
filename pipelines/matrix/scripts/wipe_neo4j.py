"""Wipes neo4j instance locally."""
from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "admin")
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    print(
        "records right now: "
        + str(len(driver.execute_query("MATCH(n) RETURN n").records))
    )
    print("wiping...")
    driver.execute_query("MATCH (n) DETACH DELETE n;")
    print(
        "total number of records left: "
        + str(len(driver.execute_query("MATCH(n) RETURN n").records))
    )
