#!/bin/bash -l
#SBATCH --job-name=fopt-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=milliways-ext
#SBATCH --array=0-2%3
#SBATCH --output=Articles/results/logs/smoke-%A_%a.out
#SBATCH --error=Articles/results/logs/smoke-%A_%a.err

# Submeta da raiz do repositório. A pasta logs deve existir antes de sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then
    module load "$FOPT_PYTHON_MODULE"
fi
: "${FOPT_RESULTS_ROOT:?Falta exportar FOPT_RESULTS_ROOT no terminal de submissão. Consulte Articles/CLUSTER_CHE.md e submeta com sbatch --export=ALL.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
printf 'Destino persistente: %s\n' "$FOPT_RESULTS_ROOT"
printf 'Python selecionado: %s\n' "$python_executable"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-smoke-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-$SLURM_JOB_ID"
# Teste 1: lógica de fronteiras e retomada, em temporários locais ao nó.
"$python_executable" -m pytest -q Articles/tests --basetemp "$local_work/unit-tests"
# O runner exige uma pasta vazia separada dos temporários do pytest.
# Teste 2: duas colunas m6, semente própria em C e gráficos no checkpoint.
"$python_executable" -m Articles.cluster_runner \
    --smoke --beta-check --batch-size 1 \
    --workers 1 --scenario-index "$SLURM_ARRAY_TASK_ID" \
    --work-dir "$local_work/campaign" \
    --output "$FOPT_RESULTS_ROOT/smoke_v3/m8-$SLURM_ARRAY_TASK_ID"
