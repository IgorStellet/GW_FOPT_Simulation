#!/usr/bin/env bash
# Uso: bash article_FOPT_combined/cluster/submit.sh STAGE [opções de sbatch]
# Exemplo: ... boundaries --partition=PARTICAO --array=0-31%8
set -euo pipefail
stage="${1:?Informe pilot, boundaries, observables ou figures.}"
shift
case "$stage" in pilot|boundaries|observables|figures) ;; *) echo 'Estágio inválido.' >&2; exit 2 ;; esac
test -f article_FOPT_combined/model.py || { echo 'Entre na raiz do checkout antes de submeter.' >&2; exit 2; }
test -n "${FOPT_RESULTS_ROOT:-}" || { echo 'Exporte FOPT_RESULTS_ROOT antes de submeter.' >&2; exit 2; }
mkdir -p article_FOPT_combined/results/logs
exec sbatch --export=ALL "$@" "article_FOPT_combined/cluster/$stage.slurm"
