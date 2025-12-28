#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"
LOCK_FILE="$PROJECT_ROOT/uv.lock"
VENV_DIR="/home/vscode/.venv"
STAMP_FILE="$VENV_DIR/.setup.stamp"

echo "== Setup: ensure home venv at $VENV_DIR =="

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip wheel setuptools

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export UV_CACHE_DIR="/home/vscode/.cache/uv"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed; aborting." >&2
  exit 1
fi

HASH_INPUT="$(cat "$PYPROJECT_FILE")"
if [ -f "$LOCK_FILE" ]; then
  HASH_INPUT="$HASH_INPUT$(cat "$LOCK_FILE")"
fi
CURRENT_HASH="$(printf "%s" "$HASH_INPUT" | sha256sum | awk '{print $1}')"

if [ -f "$STAMP_FILE" ] && grep -q "$CURRENT_HASH" "$STAMP_FILE"; then
  echo "== Dependencies already installed (pyproject/lock unchanged) =="
else
  echo "== Installing project dependencies with uv =="
  uv sync --no-dev --python "$VENV_DIR/bin/python"
  echo "== Registering ipykernel =="
  "$VENV_DIR/bin/python" -m ipykernel install --user --name "project-venv" --display-name "Python (.venv)"
  echo "$CURRENT_HASH" > "$STAMP_FILE"
fi

echo "== Setup complete =="