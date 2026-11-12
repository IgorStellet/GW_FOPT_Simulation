#!/bin/bash -l
#SBATCH --job-name=fopt-paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=milliways-ext
#SBATCH --output=Articles/results/logs/publication-figures-%j.out
#SBATCH --error=Articles/results/logs/publication-figures-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then module load "$FOPT_PYTHON_MODULE"; fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-publication-$SLURM_JOB_ID"
export TMPDIR="$local_work"
arguments=(--boundaries "${FOPT_BOUNDARIES_ROOT:-$FOPT_RESULTS_ROOT/boundaries_v1}"
    --observables "${FOPT_EDGE_DATA_ROOT:-$FOPT_RESULTS_ROOT/edge_observables_v1}"
    --reach "${FOPT_LISA_DATA_ROOT:-$FOPT_RESULTS_ROOT/lisa_reach_v1}"
    --output "${FOPT_PUBLICATION_ROOT:-$FOPT_RESULTS_ROOT/article_publication_v1}")
supplement="${FOPT_PUBLICATION_REFINEMENT_ROOT:-$FOPT_RESULTS_ROOT/publication_refinement_v1}"
if [[ -d "$supplement" ]]; then arguments+=(--observables "$supplement"); fi
if [[ "${FOPT_REPLACE_FIGURES:-0}" == 1 ]]; then arguments+=(--replace-figures); fi
"$python_executable" -m Articles.figures.publication "${arguments[@]}"
