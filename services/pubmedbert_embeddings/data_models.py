from typing import List, Literal, Union

from models import BASELINE_MODEL, CHALLENGER_MODEL
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    input: Union[str, List[str]] = Field(..., description="The input text(s) to embed.")
    model: Literal[BASELINE_MODEL, CHALLENGER_MODEL] = Field(..., description="The model to use for embedding.")


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
