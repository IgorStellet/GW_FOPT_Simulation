#!/bin/bash -l
#SBATCH --job-name=fopt-paper-thermo
#SBATCH --partition=milliways-ext
#SBATCH --array=0-15%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=14:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=article_FOPT_combined/results/logs/observables-%A_%a.out
#SBATCH --error=article_FOPT_combined/results/logs/observables-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/article_FOPT_combined/cluster/environment.sh"
refine_arguments=()
if [[ "${FOPT_REFINE:-0}" == 1 ]]; then refine_arguments+=(--refine); fi
"$python_executable" -m article_FOPT_combined observables --output "$FOPT_RESULTS_ROOT/campaign" \
    --shard-index "${SLURM_ARRAY_TASK_ID:?}" --shard-count 16 --workers "${SLURM_CPUS_PER_TASK:-1}" \
    --timeout 900 --wall-seconds "${FOPT_WALL_SECONDS:-48600}" --scratch "$local_work" \
    "${retry_arguments[@]}" "${refine_arguments[@]}"
