# Use an NVIDIA base image with CUDA and install Python 3.11
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM $CUDA_IMAGE AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    gcc curl g++ openjdk-11-jre-headless libomp-dev git && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    apt-get clean


# GPU setup
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set up Python dependencies
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -U uv

# Install Python dependencies
ADD packages/ ./packages/
RUN --mount=type=cache,target=/root/.cache uv venv && \
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