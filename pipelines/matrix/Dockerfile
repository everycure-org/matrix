ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE as runtime-environment

# install project requirements
# add basic gcc / g++ for building packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl g++ && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /tmp
COPY poetry.lock pyproject.toml ./
COPY packages packages
RUN /root/.local/bin/poetry install

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

EXPOSE 8888

CMD ["kedro", "run"]