ARG BASE_IMAGE=python:3.11-buster
FROM $BASE_IMAGE AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl g++

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl g++ openjdk-11-jre-headless && \
    apt-get clean;

# Package all python dependencies
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app
ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -U uv

# Install Python dependencies
ADD packages/ ./packages/
RUN uv pip install --system -r requirements.txt

# Cache drivers as part of image
COPY conf/base/spark.yml conf/base/spark.yml
COPY scripts/maven-deps.py maven-deps.py
RUN python maven-deps.py

ADD . .