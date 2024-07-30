from fastapi import APIRouter

router = APIRouter(prefix="/synonymize", tags=["synonymize"])


@router.get("")
async def get_canonical_curies():
    return "foo"
