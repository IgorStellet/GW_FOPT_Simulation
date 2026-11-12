#!/bin/bash -l
#SBATCH --job-name=fopt-cover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --partition=milliways-ext
#SBATCH --array=0-11%3
#SBATCH --signal=B:USR1@1800
#SBATCH --output=Articles/results/logs/coverage-%A_%a.out
#SBATCH --error=Articles/results/logs/coverage-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then module load "$FOPT_PYTHON_MODULE"; fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-coverage-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
export TMPDIR="$local_work"
arguments=(run --plan "${FOPT_COVERAGE_PLAN:-$FOPT_RESULTS_ROOT/coverage_plans/coverage_v2.json}"
    --output "${FOPT_COVERAGE_ROOT:-$FOPT_RESULTS_ROOT/coverage_observables_v2}"
    --work-dir "$local_work" --task-index "$SLURM_ARRAY_TASK_ID"
    --shards-per-scenario "${FOPT_COVERAGE_SHARDS:-4}"
    --point-timeout "${FOPT_COVERAGE_TIMEOUT:-900}")
reuse="${FOPT_COVERAGE_REUSE:-$FOPT_RESULTS_ROOT/coverage_pilot_v2}"
if [[ -d "$reuse" ]]; then arguments+=(--reuse "$reuse"); fi
if [[ -n "${FOPT_MAX_POINTS:-}" ]]; then arguments+=(--max-points "$FOPT_MAX_POINTS"); fi
if [[ "${FOPT_COVERAGE_RETRY_FAILED:-0}" == 1 ]]; then arguments+=(--retry-failed); fi
runner_pid=""
forward_stop() { if [[ -n "$runner_pid" ]]; then kill -USR1 "$runner_pid" 2>/dev/null || true; fi; }
trap forward_stop USR1 TERM INT
"$python_executable" -m Articles.figures.publication.coverage "${arguments[@]}" &
runner_pid=$!
while true; do
    if wait "$runner_pid"; then exit 0; else status=$?; fi
    if ! kill -0 "$runner_pid" 2>/dev/null; then exit "$status"; fi
done
