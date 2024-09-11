from pydantic import BaseModel, Field
from typing import List


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str


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
