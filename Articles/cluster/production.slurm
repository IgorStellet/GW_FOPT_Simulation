#!/bin/bash -l
#SBATCH --job-name=fopt-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --partition=generic
#SBATCH --array=0-99%2
#SBATCH --signal=B:USR1@300
#SBATCH --output=Articles/results/logs/scan-%A_%a.out
#SBATCH --error=Articles/results/logs/scan-%A_%a.err

# 100 partes disjuntas; no máximo 2 jobs de 4 CPUs simultâneos neste exemplo.
# Confirme partição, memória, tempo e quota com o CHE antes de submeter.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then
    module load "$FOPT_PYTHON_MODULE"
fi
: "${FOPT_RESULTS_ROOT:?Defina uma pasta persistente autorizada no CHE.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch_root="${FOPT_SCRATCH_ROOT:-/scratch/local}"
local_work="$scratch_root/$USER/fopt-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-$SLURM_JOB_ID"
# O aviso vai somente ao coordenador; workers não recebem USR1 e continuam
# o lote atual. Se o limite chegar antes de terminar, retome o checkpoint.
runner_pid=""
forward_stop() {
    if [[ -n "$runner_pid" ]]; then
        kill -USR1 "$runner_pid" 2>/dev/null || true
    fi
}
trap forward_stop USR1
# Se alterar a quantidade de partes, altere também array=0-(count-1).
"$python_executable" -m Articles.cluster_runner \
    --beta-check --batch-size 100 --workers "$SLURM_CPUS_PER_TASK" \
    --shard-count 100 --shard-index "$SLURM_ARRAY_TASK_ID" \
    --work-dir "$local_work" \
    --output "$FOPT_RESULTS_ROOT/production/part-$SLURM_ARRAY_TASK_ID" &
runner_pid=$!
# wait pode ser interrompido pelo sinal do Slurm; continue aguardando o job.
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
