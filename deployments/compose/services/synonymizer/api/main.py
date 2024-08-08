import logging
import sys
from fastapi import FastAPI

from api.modules.synonymize.routers import router as synonimze_router

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router=synonimze_router)


@app.get("/")
async def root():
    """root GET endpoint ."""
    return {"message": "Hello, world"}
