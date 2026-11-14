#!/usr/bin/env bash
# Fonte comum dos jobs; todo caminho persistente é explícito e validado.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submeta os jobs a partir da raiz do checkout.}"
test -f article_FOPT_combined/model.py || { echo 'SLURM_SUBMIT_DIR não é a raiz do projeto.' >&2; exit 2; }
module load "${FOPT_PYTHON_MODULE:-python/3.11.7-gcc-12.2.0}"
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT para uma NOVA pasta persistente desta campanha.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/article_FOPT_combined/.venv/bin/python}"
test -x "$python_executable" || { printf 'Python indisponível: %s\n' "$python_executable" >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
scratch_root="${FOPT_SCRATCH_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
test -d "$scratch_root" && test -w "$scratch_root" || { printf 'Temporário inexistente/sem escrita: %s\n' "$scratch_root" >&2; exit 2; }
local_work="$(mktemp -d -- "$scratch_root/fopt-combined-${SLURM_JOB_ID:-local}.XXXXXX")"
trap 'rmdir "$local_work" 2>/dev/null || true' EXIT
retry_arguments=()
if [[ "${FOPT_RETRY_FAILED:-0}" == 1 ]]; then retry_arguments+=(--retry-failed); fi
printf 'Python: %s\nResultados: %s\nTemporário: %s\n' "$python_executable" "$FOPT_RESULTS_ROOT" "$local_work"
