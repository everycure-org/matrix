from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

app = FastAPI()

# Load the model and tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = model_name


class EmbeddingResponse(BaseModel):
    object: str
    data: List[dict]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        # Ensure input is a list
        inputs = [request.input] if isinstance(request.input, str) else request.input

        embeddings = []
        total_tokens = 0

        for text in inputs:
            # Tokenize the input
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            total_tokens += tokens.input_ids.shape[1]

            with torch.no_grad():
                output = model(
                    **tokens, output_hidden_states=True, return_dict=True
                ).pooler_output
                embedding = output.cpu().detach().numpy().tolist()
            embeddings.append(embedding)

        # Create a response in OpenAI-like format
        response = EmbeddingResponse(
            object="list",
            data=[
                {"object": "embedding", "embedding": emb, "index": i}
                for i, emb in enumerate(embeddings)
            ],
            model=model_name,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
