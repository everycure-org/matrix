from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import torch
import os
from joblib import Memory

from typing import List

app = FastAPI()

# Load the model and tokenizer
model_name = os.getenv(
    "MODEL_NAME", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Setup joblib caching
cache_dir = f"./.cache/{model_name}"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0, mmap_mode="r")


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = model_name


class Embedding(BaseModel):
    object: str = Field(
        "embedding",
        description="The type of object returned, in this case it is an embedding.",
    )
    embedding: List[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="The index of the embedding in the input list.")


class Usage(BaseModel):
    prompt_tokens: int = Field(..., description="The number of tokens in the prompt.")
    total_tokens: int = Field(..., description="The total number of tokens used.")


class EmbeddingResponse(BaseModel):
    object: str = Field(
        "list",
        description="The type of object returned, in this case it is a list of embeddings.",
    )
    data: List[Embedding] = Field(..., description="The list of embedding vectors.")
    model: str = Field(..., description="The name of the embedding model used.")
    usage: Usage = Field(..., description="The usage statistics for the request.")


@memory.cache
async def get_embedding(text: str) -> List[float]:
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    token_count = tokens.input_ids.shape[1]

    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True).pooler_output
        embedding = output.cpu().detach().numpy().tolist()

    # we get a list of lists, but we just want the first one
    return embedding[0], token_count


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        # Ensure input is a list
        inputs = [request.input] if isinstance(request.input, str) else request.input

        embeddings = []
        total_tokens = 0

        for i, text in enumerate(inputs):
            # Tokenize the input
            embedding, token_count = await get_embedding(text)
            embeddings.append(embedding)
            total_tokens += token_count

        # Create a response in OpenAI-like format
        response = EmbeddingResponse(
            object="list",
            data=[
                Embedding(object="embedding", embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
            ],
            model=model_name,
            usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "success"}


@app.get("/health/liveness")
async def liveness_check():
    return {"status": "alive"}


@app.get("/health/readiness")
async def readiness_check():
    # Perform a simple operation to check if the model is loaded
    if tokenizer is not None:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=500, detail="Service not ready")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
