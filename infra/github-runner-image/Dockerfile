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
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    sh -lc 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list' && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli docker-compose-plugin docker-buildx-plugin containerd.io && \
    # Install Gcloud
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    chmod +x setup_pyenv.sh && \
    ./setup_pyenv.sh && \
    rm -f setup_pyenv.sh && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx && systemctl enable docker && systemctl start docker && sudo usermod -aG docker runner

USER runner