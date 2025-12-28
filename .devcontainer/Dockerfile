FROM mcr.microsoft.com/devcontainers/python:3.12

USER root
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl git build-essential pkg-config ca-certificates; \
    rm -rf /var/lib/apt/lists/*

USER vscode
ENV PATH="/home/vscode/.local/bin:${PATH}"
RUN set -eux; \
    mkdir -p /home/vscode/.local/bin; \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/vscode/.bashrc; \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/vscode/.profile

# Install uv with retry
RUN set -eux; \
  for i in 1 2 3; do \
    curl -LsSf https://astral.sh/uv/install.sh | sh && break || sleep 3; \
  done; \
  command -v uv && ln -sf "$(command -v uv)" /usr/local/bin/uv || true
