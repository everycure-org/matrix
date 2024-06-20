ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE as build
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl g++

# package all python dependencies
ENV PIP_NO_CACHE_DIR=1
ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -U uv && \
    uv venv
ENV PATH="/.venv/bin:$PATH"
ADD packages/ ./packages/
RUN uv pip install -r requirements.txt


# Stage 2
FROM $BASE_IMAGE

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -m -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

USER kedro
WORKDIR /home/kedro
COPY --from=build .venv .
ENV PATH="/home/kedro/.venv/bin:$PATH"
ADD . .

# EXPOSE 8888
CMD ["kedro", "run"]