# Dockerfile
FROM ghcr.io/actions/actions-runner:latest

USER root

COPY setup_pyenv.sh .

RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg make build-essential libssl-dev zlib1g-dev \
          libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
          libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
          libffi-dev liblzma-dev openjdk-17-jdk openjdk-17-jre && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/static/stable/$(uname -m)/docker-$(curl -s https://api.github.com/repos/moby/moby/releases/latest | grep tag_name | cut -d '"' -f4 | sed 's/v//').tgz \
  | tar xz --strip-components=1 -C /usr/local/bin docker/docker && \
    curl -L "https://github.com/docker/compose/releases/download/2.39.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/bin/docker-compose && \
    chmod +x /usr/bin/docker-compose && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    chmod +x setup_pyenv.sh && \
    ./setup_pyenv.sh && \
    rm -f setup_pyenv.sh && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx && usermod -aG docker runner

USER runner