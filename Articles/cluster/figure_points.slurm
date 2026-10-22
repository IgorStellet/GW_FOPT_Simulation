#!/bin/bash -l
#SBATCH --job-name=fopt-paper-points
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --partition=milliways-ext
#SBATCH --array=0-2%3
#SBATCH --signal=B:USR1@1800
#SBATCH --output=Articles/results/logs/paper-points-%A_%a.out
#SBATCH --error=Articles/results/logs/paper-points-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then module load "$FOPT_PYTHON_MODULE"; fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-paper-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-$SLURM_JOB_ID"
arguments=(points --plan "${FOPT_POINT_PLAN:-$FOPT_RESULTS_ROOT/article_figures_v1/observables_plan.json}"
    --scenario-index "$SLURM_ARRAY_TASK_ID" --work-dir "$local_work"
    --output "${FOPT_SELECTED_ROOT:-$FOPT_RESULTS_ROOT/article_points_v1}/m8-$SLURM_ARRAY_TASK_ID"
    --point-timeout "${FOPT_SELECTED_TIMEOUT:-900}")
if [[ -n "${FOPT_MAX_POINTS:-}" ]]; then arguments+=(--max-points "$FOPT_MAX_POINTS"); fi
if [[ "${FOPT_RETRY_SELECTED:-0}" == 1 ]]; then arguments+=(--retry-failed); fi
runner_pid=""
forward_stop() { if [[ -n "$runner_pid" ]]; then kill -USR1 "$runner_pid" 2>/dev/null || true; fi; }
trap forward_stop USR1
"$python_executable" -m Articles.figures "${arguments[@]}" &
runner_pid=$!
while true; do
    if wait "$runner_pid"; then exit 0; else status=$?; fi
    if ! kill -0 "$runner_pid" 2>/dev/null; then exit "$status"; fi
done
