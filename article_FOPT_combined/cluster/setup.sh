#!/bin/bash -l
# Execute com bash a partir da raiz do checkout; não altera o ambiente da campanha antiga.
set -euo pipefail
project_root="$(pwd -P)"
test -f "$project_root/article_FOPT_combined/model.py" || { echo 'Entre na raiz do repositório.' >&2; exit 2; }
if [[ -n "${FOPT_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}" ]]; then
    module load "${FOPT_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
fi
python3 -c 'import sys; assert sys.version_info >= (3,11), "Use Python >=3.11"; print(sys.executable, sys.version)'
python3 -m venv "$project_root/article_FOPT_combined/.venv"
python_executable="$project_root/article_FOPT_combined/.venv/bin/python"
"$python_executable" -m ensurepip --upgrade
"$python_executable" -m pip install -r "$project_root/article_FOPT_combined/requirements.txt"
"$python_executable" -B -m pytest -c article_FOPT_combined/pytest.ini article_FOPT_combined/tests -q
mkdir -p "$project_root/article_FOPT_combined/results/logs"
printf '\nAmbiente pronto. Nas próximas chamadas use:\nexport FOPT_PYTHON="%s"\n' "$python_executable"
