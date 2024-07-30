from fastapi import APIRouter
from pydantic import BaseModel

from .node_synonymizer import NodeSynonymizer

router = APIRouter(prefix="/synonymize", tags=["synonymize"])

synonymizer = NodeSynonymizer()


class Search(BaseModel):
    """Model to searches."""

    name: str


@router.get("")
async def get_canonical_curies(search: Search):
    """Endpoint to get canonical curies.

    Args:
        Searhc: search string
    """
    return synonymizer.get_canonical_curies(names=[search.name])
