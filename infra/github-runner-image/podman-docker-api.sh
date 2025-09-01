#!/usr/bin/env bash
set -euo pipefail

# Rootless podman needs these
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/run}"
mkdir -p "$XDG_RUNTIME_DIR" "$XDG_RUNTIME_DIR/podman"

# Start Podman’s Docker-compatible REST service (no timeout)
# It will listen on: unix://$XDG_RUNTIME_DIR/podman/podman.sock
if ! pgrep -f "podman system service" >/dev/null 2>&1; then
  nohup podman system service --time=0 "unix://$XDG_RUNTIME_DIR/podman/podman.sock" \
    >/tmp/podman-service.log 2>&1 &
  # Give it a moment to create the socket
  for i in {1..20}; do
    [[ -S "$XDG_RUNTIME_DIR/podman/podman.sock" ]] && break
    sleep 0.15
  done
fi

# Sanity: show what the docker client is targeting (optional)
# docker context ls || true

exec "$@"