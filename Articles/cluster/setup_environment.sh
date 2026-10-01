#!/usr/bin/env bash
# Prepara o ambiente no nó de entrada; não submete jobs nem calcula bounces.
# Uso: bash Articles/cluster/setup_environment.sh [PYTHON] [PASTA_NOVA_DA_VENV]
set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Uso: bash Articles/cluster/setup_environment.sh [PYTHON] [PASTA_NOVA_DA_VENV]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
python_executable="${1:-python3}"
environment_dir="${2:-$repo_root/.venv-che311}"

# O Python base pode não ter pip. venv + ensurepip instalam-no no ambiente
# privado, sem modificar /opt/spack ou depender de um pip de outro Python.
"$python_executable" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Exige Python >=3.11; carregue o módulo correto antes de continuar.")
print("Python escolhido:", sys.executable, sys.version.split()[0])
import venv, ensurepip
' || {
    echo "O Python escolhido precisa oferecer venv e ensurepip; consulte os módulos/suporte do CHE." >&2
    exit 1
}

if [[ -e "$environment_dir" || -L "$environment_dir" ]]; then
    echo "A pasta já existe: $environment_dir" >&2
    echo "Escolha outra pasta NOVA no segundo argumento; o ambiente anterior foi preservado." >&2
    exit 2
fi

"$python_executable" -m venv --without-pip "$environment_dir"
environment_python="$environment_dir/bin/python"
"$environment_python" -m ensurepip --upgrade
"$environment_python" -m pip install --upgrade pip
"$environment_python" -m pip install -e "$repo_root" pytest
"$environment_python" -m pip check
"$environment_python" -m Articles.collect_data --dry-run --output "$environment_dir/installation-check"

echo "Ambiente pronto. Use este Python para os testes e para os jobs:"
printf 'export FOPT_PYTHON=%q\n' "$(cd -- "$environment_dir" && pwd)/bin/python"
echo '"$FOPT_PYTHON" -m pytest -q'
echo "Os testes numéricos devem rodar em uma alocação Slurm. Consulte Articles/CLUSTER_CHE.md."
