#!/bin/bash -l
#SBATCH --job-name=fopt-paper-pilot
#SBATCH --partition=milliways-ext
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=article_FOPT_combined/results/logs/pilot-%A_%a.out
#SBATCH --error=article_FOPT_combined/results/logs/pilot-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/article_FOPT_combined/cluster/environment.sh"
masses=(inf 840.8964152537145 668.740304976422 400)
couplings=(3.5 3.4 3.3 0.0)
index="${SLURM_ARRAY_TASK_ID:?}"
wall_seconds="${FOPT_WALL_SECONDS:-6600}"
case "${FOPT_PILOT_STAGE:-point}" in
    point)
        "$python_executable" -m article_FOPT_combined point --output "$FOPT_RESULTS_ROOT/pilot" \
            --m6 inf --m8 "${masses[$index]}" --C "${couplings[$index]}" \
            --timeout 1200 --wall-seconds "$wall_seconds" --scratch "$local_work" "${retry_arguments[@]}"
        ;;
    boundaries)
        "$python_executable" -m article_FOPT_combined boundaries --output "$FOPT_RESULTS_ROOT/pilot" \
            --shard-index "$index" --shard-count 4 --timeout "${FOPT_BOUNDARY_TIMEOUT:-300}" \
            --wall-seconds "$wall_seconds" \
            --scratch "$local_work" "${retry_arguments[@]}"
        ;;
    observables)
        "$python_executable" -m article_FOPT_combined observables --output "$FOPT_RESULTS_ROOT/pilot" \
            --shard-index "$index" --shard-count 4 --workers 1 --timeout 900 --wall-seconds "$wall_seconds" \
            --scratch "$local_work" "${retry_arguments[@]}"
        ;;
    *) printf 'FOPT_PILOT_STAGE deve ser point, boundaries ou observables.\n' >&2; exit 2 ;;
esac
