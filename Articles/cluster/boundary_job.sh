#!/usr/bin/env bash
# Entrypoint compartilhado dos quatro templates novos. Não altera o fluxo v3.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then
    module load "$FOPT_PYTHON_MODULE"
fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT e submeta com sbatch --export=ALL.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-boundaries-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-$SLURM_JOB_ID"
stage="${FOPT_STAGE:-boundaries}"
pilot="${FOPT_PILOT:-0}"
if [[ "$stage" != boundaries && "$stage" != interior ]]; then
    printf 'FOPT_STAGE deve ser boundaries ou interior\n' >&2
    exit 2
fi
if [[ "$pilot" == 1 ]]; then
    suffix="${stage}_pilot_v1"
    "$python_executable" -m pytest -q Articles/tests
else
    suffix="${stage}_v1"
fi
arguments=(--stage "$stage" --scenario-index "$SLURM_ARRAY_TASK_ID"
    --output "$FOPT_RESULTS_ROOT/$suffix/m8-$SLURM_ARRAY_TASK_ID" --work-dir "$local_work"
    --point-timeout "${FOPT_POINT_TIMEOUT:-300}" --criterion "${FOPT_BOUNDARY_CRITERION:-strong}"
    --C-tolerance "${FOPT_C_TOLERANCE:-0.02}" --plots)
if [[ "$pilot" == 1 ]]; then
    arguments+=(--smoke)
fi
if [[ -n "${FOPT_REUSE_ROOT:-}" ]]; then
    arguments+=(--reuse "$FOPT_REUSE_ROOT")
fi
if [[ "$stage" == interior ]]; then
    if [[ "$pilot" == 1 ]]; then
        boundary_default="$FOPT_RESULTS_ROOT/boundaries_pilot_v1"
    else
        boundary_default="$FOPT_RESULTS_ROOT/boundaries_v1"
    fi
    arguments+=(--boundaries "${FOPT_BOUNDARIES_ROOT:-$boundary_default}" --beta-check)
    if [[ "${FOPT_ALLOW_PARTIAL_BOUNDARIES:-0}" == 1 ]]; then
        arguments+=(--allow-partial-boundaries)
    fi
fi
printf 'Etapa: %s; piloto: %s; Python: %s\n' "$stage" "$pilot" "$python_executable"
printf 'Destino persistente: %s\n' "$FOPT_RESULTS_ROOT/$suffix/m8-$SLURM_ARRAY_TASK_ID"
runner_pid=""
forward_stop() {
    if [[ -n "$runner_pid" ]]; then
        kill -USR1 "$runner_pid" 2>/dev/null || true
    fi
}
trap forward_stop USR1
"$python_executable" -m Articles.boundary_scan "${arguments[@]}" &
runner_pid=$!
while true; do
    if wait "$runner_pid"; then
        exit 0
    else
        status=$?
    fi
    if ! kill -0 "$runner_pid" 2>/dev/null; then
        exit "$status"
    fi
done
