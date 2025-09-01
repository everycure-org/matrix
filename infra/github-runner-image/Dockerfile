FROM ghcr.io/actions/actions-runner:latest

USER root

# Core build tools you already had; add Podman rootless deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget make build-essential \
      uidmap slirp4netns fuse-overlayfs podman && \
    rm -rf /var/lib/apt/lists/*

# --- Docker CLI (client-only) ---
ARG DOCKER_CLI_VERSION=27.2.0
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64)  dl_arch="x86_64" ;; \
      aarch64) dl_arch="aarch64" ;; \
      *) echo "Unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://download.docker.com/linux/static/stable/${dl_arch}/docker-${DOCKER_CLI_VERSION}.tgz" \
    | tar xz --strip-components=1 -C /usr/bin docker/docker

# Compose v2 plugin (standalone binary also works)
ARG COMPOSE_VERSION=2.29.7
RUN curl -L "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
      -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose && \
    mkdir -p /usr/libexec/docker/cli-plugins && \
    ln -s /usr/local/bin/docker-compose /usr/libexec/docker/cli-plugins/docker-compose

# Optional: make Podman default to fuse-overlayfs in rootless
RUN mkdir -p /etc/containers && \
    printf '[storage]\ndriver="overlay"\n[engine]\n' > /etc/containers/storage.conf && \
    printf 'events_logger="file"\n' >> /etc/containers/containers.conf

# Entrypoint: start Podman’s Docker-API service, then exec CMD
COPY --chown=runner:runner podman-docker-api.sh /usr/local/bin/podman-docker-api.sh
RUN chmod +x /usr/local/bin/podman-docker-api.sh

# Ensure runner can create the runtime dir
ENV XDG_RUNTIME_DIR=/tmp/run
RUN mkdir -p /tmp/run && chown -R runner:runner /tmp/run

USER runner
ENV DOCKER_HOST=unix:///tmp/run/podman/podman.sock

ENTRYPOINT ["/usr/local/bin/podman-docker-api.sh"]
# Example default: print versions (override in your job)
CMD ["bash", "-lc", "docker version && docker compose version"]

USER runner