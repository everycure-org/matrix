from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from api.configuration import settings
from .node_synonymizer import NodeSynonymizer

router = APIRouter(prefix="", tags=["synonymize"])

synonymizer = NodeSynonymizer()

from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    settings.NEO4J_HOST, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)


class Search(BaseModel):
    """Model to searches."""

    name: str


@router.get("/synonymize")
async def get_canonical_curies(search: Search):
    """Endpoint to get canonical curies.

    Args:
        Searhc: search string
    """
    return synonymizer.get_canonical_curies(names=[search.name])


@router.get("/normalize")
async def get_normalizer_result(search: Search):
    """Endpoint to get canonical curies.

    Args:
        Searhc: search string
    """
    result = synonymizer.get_normalizer_results(entities=[search.name])

    if result.get(search.name) and check_node_exists(
        result[search.name]["id"]["identifier"], "Entity"
    ):
        return result

    return {}


def check_node_exists(id: str, label: str):
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        query = f"""
        MATCH (n:{label} {{id: $id}})
        RETURN count(n) > 0 AS node_exists
        """
        result = session.run(query, id=id)
        return result.single()["node_exists"]
