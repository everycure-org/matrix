"""Wipes neo4j instance locally."""
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

if os.environ.get("NEO4J_PASSWORD", "") is not "":
    print(
        "careful, you may be accidentally wiping a remote neo, saving you from this now!"
    )
    exit()


with GraphDatabase.driver(
    os.environ["NEO4J_HOST"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
) as driver:
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
