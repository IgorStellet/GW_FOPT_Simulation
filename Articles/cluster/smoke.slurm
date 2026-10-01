#!/bin/bash -l
#SBATCH --job-name=fopt-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=debug
#SBATCH --output=Articles/results/logs/smoke-%j.out
#SBATCH --error=Articles/results/logs/smoke-%j.err

# Submeta da raiz do repositório. A pasta logs deve existir antes de sbatch.
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
local_work="$scratch_root/$USER/fopt-smoke-$SLURM_JOB_ID"
"$python_executable" -m Articles.cluster_runner \
    --smoke --beta-check --batch-size 1 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --work-dir "$local_work" --output "$FOPT_RESULTS_ROOT/smoke"
