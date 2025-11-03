#!/usr/bin/env bash
set -euo pipefail

# Run from repo root. Creates .venv and installs dev deps.
if [ ! -f "pyproject.toml" ]; then
  echo "Run this from the repository root (where pyproject.toml lives)."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel
# Prefer dev extras; fall back gracefully if not defined yet
if ! pip install -e ".[dev]"; then
  echo "Dev extras not found; installing base package only."
  pip install -e .
fi

# Optional: docs extras (comment out if you don't want them)
if pip install -e ".[docs]" >/dev/null 2>&1; then
  echo "Docs extras installed."
fi

# Pre-commit if available in dev extras
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install
fi

echo
echo "✅ Dev environment ready."
echo "To activate: source $VENV_DIR/bin/activate"
