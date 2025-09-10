"""This is a simple API for generating embeddings using PubMedBERT.
It exposes two models: A baseline model that was used in a previous publication
and a model that we think actually is correct, namely a sentence embedding model.

Note this is a temporary solution and we intend to move to a more generic solution.
"""

import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from data_models import Embedding, EmbeddingRequest, EmbeddingResponse, Usage
from fastapi import FastAPI, HTTPException
from models import ModelStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CPU_COUNT = multiprocessing.cpu_count()
app = FastAPI()
model_store = ModelStore()

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=CPU_COUNT)  # Adjust the number of workers as needed


async def get_embedding(texts: List[str], model_name: str) -> List[float]:
    """Async call to get embedding using the pipeline API of HF."""
    # manually truncate the text to 512 tokens
    texts = [text[:512] for text in texts]
    # Run the call in a separate thread
    return await asyncio.get_event_loop().run_in_executor(
        executor, lambda: (model_store.get_embeddings(texts, model_name), 0)
    )


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """Main API endpoint for embedding generation."""
    try:
        start_time = time.time()
        # Ensure input is a list
        inputs = [request.input] if isinstance(request.input, str) else request.input

        # Process all inputs concurrently
        if len(inputs) > 0:
            embeddings, token_count = await get_embedding(inputs, request.model)
        else:
            embeddings = []
            token_count = 0

        # Create a response in OpenAI-like format
        response = EmbeddingResponse(
            object="list",
            data=[Embedding(object="embedding", embedding=emb, index=i) for i, emb in enumerate(list(embeddings))],
            model=request.model,
            usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
        )

        logger.info(
            f"Generated embeddings for {len(inputs)} inputs in {time.time() - start_time:.2f} seconds and model {request.model}"
        )

        return response
    except Exception as e:
        # print exception and stack trace
        import traceback

        print(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    # Perform a simple operation to check if the model is loaded
    if model_store is not None:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=500, detail="Service not ready")


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000, limit_concurrency=CPU_COUNT*4, backlog=CPU_COUNT*16)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        backlog=CPU_COUNT * 64,
        log_level="info",
        access_log=True,
    )
