# Dockerfile for GitHub Actions Runner Set. Used for running github actions on self hosted kubernetes.
FROM ghcr.io/actions/actions-runner:latest

USER root

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y ca-certificates curl gnupg make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev openjdk-17-jdk openjdk-17-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && uv python install

USER runner