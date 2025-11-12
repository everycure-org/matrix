ARG CUDA_IMAGE=ubuntu:24.04
FROM $CUDA_IMAGE AS build

ENV UV_LINK_MODE=copy

# ---------------------------------
# ------- GPU/System start ------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc curl g++ openjdk-17-jre-headless libomp-dev git && \
    apt-get clean

# ------- GPU/System end ----------
# ---------------------------------
# ------- Dependencies start ------
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

WORKDIR /matrix
COPY pyproject.toml uv.lock ./

# now we run this inside of the matrix pipeline folder.
WORKDIR /matrix/pipelines/matrix
COPY pipelines/matrix/pyproject.toml .
# We use --no-install-workspace to avoid re-installing the the entire workspace
# dependncies here, only installing the matrix pipeline dependencies
RUN --mount=type=cache,target=/root/.cache uv sync --frozen  --no-install-workspace
ENV PATH=/matrix/.venv/bin:$PATH

# Cache spark drivers as part of image
COPY pipelines/matrix/conf/base/spark.yml ./conf/base/spark.yml
COPY pipelines/matrix/scripts/maven-deps.py ./maven-deps.py
RUN --mount=type=cache,target=/root/.ivy2/cache python maven-deps.py

# We copy in the libs folder here and run a full sync which leads to only the delta
# being installed which are... just the libs. the matrix pipeline 
# pyproject.toml explicitly defines these as dependencies, thus it tries installing them now
COPY libs/ ../../libs/
RUN uv sync --frozen
# ---------- Dependencies end -----
# ---------------------------------

COPY pipelines/matrix/ .

# Set git commit sha as env variable LAST to avoid cache invalidation
# This should be the final layer that changes per commit
ARG GIT_SHA
ENV GIT_SHA=$GIT_SHA
