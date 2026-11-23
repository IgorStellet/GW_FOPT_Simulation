#!/bin/bash -l
#SBATCH --job-name=fopt-paper-edges
#SBATCH --partition=milliways-ext
#SBATCH --array=0-31%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=14:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=article_FOPT_combined/results/logs/boundaries-%A_%a.out
#SBATCH --error=article_FOPT_combined/results/logs/boundaries-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/article_FOPT_combined/cluster/environment.sh"
"$python_executable" -m article_FOPT_combined boundaries --output "$FOPT_RESULTS_ROOT/campaign" \
    --shard-index "${SLURM_ARRAY_TASK_ID:?}" --shard-count 32 --timeout "${FOPT_BOUNDARY_TIMEOUT:-300}" \
    --wall-seconds "${FOPT_WALL_SECONDS:-48600}" --scratch "$local_work" "${retry_arguments[@]}"
