#!/bin/bash -l
#SBATCH --job-name=fopt-paper-figures
#SBATCH --partition=milliways-ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=article_FOPT_combined/results/logs/figures-%j.out
#SBATCH --error=article_FOPT_combined/results/logs/figures-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/article_FOPT_combined/cluster/environment.sh"
figure_arguments=()
if [[ "${FOPT_DRAFT:-0}" == 1 ]]; then figure_arguments+=(--draft); fi
if [[ "${FOPT_REPLACE_FIGURES:-0}" == 1 ]]; then figure_arguments+=(--replace); fi
"$python_executable" -m article_FOPT_combined figures --output "$FOPT_RESULTS_ROOT/campaign" \
    --destination "$FOPT_RESULTS_ROOT/figures" "${figure_arguments[@]}"
