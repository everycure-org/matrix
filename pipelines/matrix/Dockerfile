ARG CUDA_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
FROM $CUDA_IMAGE AS build

# ---------------------------------
# ------- GPU/System start ------
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
LABEL com.nvidia.volumes.needed="nvidia_driver"
# ------- GPU/System end ----------
# ---------------------------------
# ------- Dependencies start ------
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

WORKDIR /matrix
COPY pyproject.toml uv.lock ./

# now we run this inside of the matrix pipeline folder.
WORKDIR /matrix/pipelines/matrix
COPY pipelines/matrix/pyproject.toml .
RUN --mount=type=cache,target=/root/.cache uv sync --frozen  --no-install-workspace
ENV PATH=/matrix/.venv/bin:$PATH

# Cache spark drivers as part of image
COPY pipelines/matrix/conf/base/spark.yml ./conf/base/spark.yml
COPY pipelines/matrix/scripts/maven-deps.py ./maven-deps.py
RUN --mount=type=cache,target=/root/.ivy2/cache python maven-deps.py

# bring in the dependent libraries and install those too
COPY libs/ ../../libs/
RUN uv sync --frozen
# ---------- Dependencies end -----
# ---------------------------------

COPY pipelines/matrix/ .

# Set git commit sha as env variable LAST to avoid cache invalidation
# This should be the final layer that changes per commit
ARG GIT_SHA
ENV GIT_SHA=$GIT_SHA
