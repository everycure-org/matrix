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

# ------- Dependencies start ------
# ---------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

WORKDIR /matrix
COPY pyproject.toml uv.lock ./

# we copy all workspace dependencies over for uv sync to work
COPY libs/ libs/
COPY docs/pyproject.toml docs/pyproject.toml

# now we run this inside of the matrix pipeline folder. This only installs 
# deps for this project, not e.g. docs
WORKDIR /matrix/pipelines/matrix
COPY pipelines/matrix/pyproject.toml .

RUN echo 'this is a test, that should always be cached'
RUN --mount=type=cache,target=/root/.cache uv sync --frozen --no-install-project

# enables kedro & python to be in the path
ENV PATH=/matrix/.venv/bin:$PATH
# Cache drivers as part of image
COPY pipelines/matrix/conf/base/spark.yml ./conf/base/spark.yml
COPY pipelines/matrix/scripts/maven-deps.py ./maven-deps.py

RUN --mount=type=cache,target=/root/.ivy2/cache python maven-deps.py
# ---------- Dependencies end -----
# ---------------------------------

# finally bring in the code of the pipeline
COPY pipelines/matrix/ .

# Set git commit sha as env variable LAST to avoid cache invalidation
# This should be the final layer that changes per commit
ARG GIT_SHA
ENV GIT_SHA=$GIT_SHA

LABEL com.nvidia.volumes.needed="nvidia_driver"