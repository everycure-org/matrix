# Dockerfile
FROM ghcr.io/actions/actions-runner:latest

USER root

COPY setup_pyenv.sh .

RUN apt-get update && \
    apt-get install -y ca-certificates curl gnupg make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev openjdk-17-jdk openjdk-17-jre && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    sh -lc 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list' && \
    apt-get update && \
    apt-get install -y docker-ce-cli docker-compose-plugin docker-buildx-plugin containerd.io && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && ./setup_pyenv.sh

USER runner