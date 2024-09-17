ARG BASE_IMAGE=python:3.11-buster
# FUTURE: switch to slim image
# ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl g++

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl g++ openjdk-11-jre-headless && \
    apt-get clean;

# Package all python dependencies
ENV PIP_NO_CACHE_DIR=1
# FUTURE avoid caching
#ENV UV_NO_CACHE=true

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
ADD gcs-connector-hadoop3-2.2.2-shaded.jar .

# Set git commit sha as env variable, useful for tracing back code to data
ARG GIT_SHA
ENV GIT_SHA=$GIT_SHA

ADD . .