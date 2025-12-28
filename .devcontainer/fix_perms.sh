#!/usr/bin/env bash
set -euo pipefail
# Ensure that the vscode user owns its pip and uv cache directories
sudo mkdir -p /home/vscode/.cache/pip /home/vscode/.cache/uv

sudo chown -R vscode:vscode /home/vscode/.cache
