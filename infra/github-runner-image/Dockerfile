# Dockerfile for GitHub Actions Runner Set. Used for running github actions on self hosted kubernetes.
FROM ghcr.io/actions/actions-runner:2.332.0

USER root

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y ca-certificates curl gnupg make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev openjdk-17-jdk openjdk-17-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && uv python install

# Install Docker Compose V2 plugin.
# Map dpkg architecture names to Compose release asset names and fail on HTTP errors.
RUN set -eux; \
    DOCKER_CONFIG=${DOCKER_CONFIG:-/usr/local/lib/docker}; \
    mkdir -p "$DOCKER_CONFIG/cli-plugins"; \
    ARCH=$(dpkg --print-architecture); \
    case "$ARCH" in \
      amd64) COMPOSE_ARCH="x86_64" ;; \
      arm64) COMPOSE_ARCH="aarch64" ;; \
      armhf) COMPOSE_ARCH="armv7" ;; \
      *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;; \
    esac; \
    curl -fSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${COMPOSE_ARCH}" \
      -o "$DOCKER_CONFIG/cli-plugins/docker-compose"; \
    chmod +x "$DOCKER_CONFIG/cli-plugins/docker-compose"; \
    "$DOCKER_CONFIG/cli-plugins/docker-compose" version

USER runner