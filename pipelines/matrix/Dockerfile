ARG CUDA_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
FROM $CUDA_IMAGE AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc curl g++ openjdk-17-jre-headless libomp-dev git && \
    apt-get clean


# GPU setup
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD requirements.txt .

# Install Python dependencies
ADD packages/ ./packages/
RUN --mount=type=cache,target=/root/.cache uv venv -p 3.11 && \
    uv pip install -r requirements.txt
ENV PATH=/app/.venv/bin:$PATH

# Cache drivers as part of image
COPY conf/base/spark.yml conf/base/spark.yml
COPY scripts/maven-deps.py maven-deps.py
RUN python maven-deps.py

# Set git commit sha as env variable, useful for tracing back code to data
ARG GIT_SHA
ENV GIT_SHA=$GIT_SHA

ADD . .

LABEL com.nvidia.volumes.needed="nvidia_driver"