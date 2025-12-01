#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Dev setup for GW_FOPT_Simulation
#
# - Cria um venv em .venv
# - Instala o pacote em modo editável a partir de pyproject.toml
# - Instala extras [dev] e opcionalmente [docs]
# - Garante que src/ seja a primeira entrada relevante no PYTHONPATH
#   (legacy/ NÃO entra no caminho de import em nenhum momento)
# ---------------------------------------------------------------------------

if [ ! -f "pyproject.toml" ]; then
  echo "Run this from the repository root (where pyproject.toml lives)."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Using PYTHON_BIN = $PYTHON_BIN"
echo "Creating/using virtualenv in $VENV_DIR"
echo

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Coloca src/ como primeira entrada extra no caminho de imports
# (isso ajuda a rodar scripts de docs/examples sem precisar mexer em sys.path)
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m pip install -U pip wheel

# ---------------------------------------------------------------------------
# Instalação do pacote com extras
# ---------------------------------------------------------------------------

# Por padrão, tentamos instalar com extras de dev.
# Se quiser customizar, você pode chamar, por exemplo:
#   EXTRAS="dev,docs" ./scripts/dev_setup.sh
EXTRAS="${EXTRAS:-dev}"

echo "Installing package in editable mode with extras: [${EXTRAS}]"
if ! pip install -e ".[${EXTRAS}]"; then
  echo "Extras [${EXTRAS}] not found; installing base package only."
  pip install -e .
fi

# Tenta docs extras silenciosamente, se existirem
if pip install -e ".[docs]" >/dev/null 2>&1; then
  echo "Docs extras installed."
fi

# Pre-commit se estiver disponível no ambiente
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install
fi

echo
echo "✅ Dev environment ready."
echo "To activate: source $VENV_DIR/bin/activate"
echo "Python will import CosmoTransitions from src/ (legacy/ is ignored)."
