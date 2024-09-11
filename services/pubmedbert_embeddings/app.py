from fastapi import FastAPI, HTTPException
from transformers import pipeline
import os
from joblib import Memory
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from models import EmbeddingRequest, EmbeddingResponse, Usage, Embedding

import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

app = FastAPI()

# Load the model and tokenizer using a pipeline
model_name = os.getenv(
    "MODEL_NAME", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
feature_extractor = pipeline(
    "feature-extraction", model=model_name, tokenizer=model_name
)

# Setup joblib caching
cache_dir = f"./.cache/{model_name}"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0, mmap_mode="r")

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as needed


# @memory.cache
async def get_embedding(texts: List[str]) -> List[float]:
    """Async call to get embedding using the pipeline API of HF."""

    def run_pipeline():
        features = feature_extractor(
            texts, padding=True, truncation=True, max_length=512
        )
        embedding = features[0][0]  # Get the CLS token embedding
        token_count = len(feature_extractor.tokenizer.encode(" ".join(texts)))
        return embedding, token_count

    # Run the pipeline in a separate thread
    return await asyncio.get_event_loop().run_in_executor(executor, run_pipeline)


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """Main API endpoint for embedding generation."""
    try:
        # Ensure input is a list
        inputs = [request.input] if isinstance(request.input, str) else request.input

        # Process all inputs concurrently
        if len(inputs) > 0:
            embeddings, token_count = await get_embedding(inputs)
        else:
            embeddings = []
            token_count = 0

        # Create a response in OpenAI-like format
        response = EmbeddingResponse(
            object="list",
            data=[
                Embedding(object="embedding", embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=model_name,
            usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    # Perform a simple operation to check if the model is loaded
    if feature_extractor is not None:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=500, detail="Service not ready")


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000, limit_concurrency=CPU_COUNT*4, backlog=CPU_COUNT*16)
    uvicorn.run(app, host="0.0.0.0", port=8000, backlog=CPU_COUNT * 64)
