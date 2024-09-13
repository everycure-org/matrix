ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ADD requirements.txt .

RUN uv venv && \
    uv pip install -r requirements.txt
ENV PATH=/app/.venv/bin:$PATH
# Download and cache the model
ADD models.py .
RUN python -c "from models import ModelStore; \
    model_store = ModelStore()"

ADD . .


CMD "python" "app.py"