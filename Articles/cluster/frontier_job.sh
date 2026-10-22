#!/usr/bin/env bash
# Compartilha apenas a preparação e a propagação de sinais dos dois jobs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then module load "$FOPT_PYTHON_MODULE"; fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-$FOPT_FRONTIER_STAGE-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
export TMPDIR="$local_work"
case "$FOPT_FRONTIER_STAGE" in
    edges)
        plan="${FOPT_EDGE_PLAN:-$FOPT_RESULTS_ROOT/frontier_plans/edges_v1.json}"
        output="${FOPT_EDGE_DATA_ROOT:-$FOPT_RESULTS_ROOT/edge_observables_v1}"
        command=run-edges
        ;;
    lisa)
        plan="${FOPT_LISA_PLAN:-$FOPT_RESULTS_ROOT/frontier_plans/lisa_v1.json}"
        output="${FOPT_LISA_DATA_ROOT:-$FOPT_RESULTS_ROOT/lisa_reach_v1}"
        command=run-lisa
        ;;
    *) printf 'Etapa inválida: %s\n' "$FOPT_FRONTIER_STAGE" >&2; exit 2 ;;
esac
arguments=("$command" --plan "$plan" --output "$output" --work-dir "$local_work"
    --task-index "$SLURM_ARRAY_TASK_ID" --shards-per-scenario "${FOPT_FRONTIER_SHARDS:-4}"
    --point-timeout "${FOPT_FRONTIER_TIMEOUT:-900}")
if [[ -n "${FOPT_MAX_POINTS:-}" ]]; then arguments+=(--max-points "$FOPT_MAX_POINTS"); fi
if [[ "${FOPT_FRONTIER_RETRY_FAILED:-0}" == 1 ]]; then arguments+=(--retry-failed); fi
if [[ "$FOPT_FRONTIER_STAGE" == lisa ]]; then
    reuse="${FOPT_EDGE_DATA_ROOT:-$FOPT_RESULTS_ROOT/edge_observables_v1}"
    if [[ -d "$reuse" ]]; then arguments+=(--reuse "$reuse"); fi
fi
runner_pid=""
forward_stop() { if [[ -n "$runner_pid" ]]; then kill -USR1 "$runner_pid" 2>/dev/null || true; fi; }
trap forward_stop USR1 TERM INT
"$python_executable" -m Articles.figures.frontiers "${arguments[@]}" &
runner_pid=$!
while true; do
    if wait "$runner_pid"; then exit 0; else status=$?; fi
    if ! kill -0 "$runner_pid" 2>/dev/null; then exit "$status"; fi
done
