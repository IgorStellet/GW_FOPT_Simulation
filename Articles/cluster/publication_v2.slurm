#!/bin/bash -l
#SBATCH --job-name=fopt-paper-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=milliways-ext
#SBATCH --output=Articles/results/logs/publication-v2-%j.out
#SBATCH --error=Articles/results/logs/publication-v2-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then module load "$FOPT_PYTHON_MODULE"; fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-paper-v2-$SLURM_JOB_ID"
export TMPDIR="$local_work"
arguments=(--boundaries "$FOPT_RESULTS_ROOT/boundaries_v1"
    --observables "$FOPT_RESULTS_ROOT/edge_observables_v1"
    --reach "$FOPT_RESULTS_ROOT/lisa_reach_v1"
    --output "${FOPT_PUBLICATION_V2_ROOT:-$FOPT_RESULTS_ROOT/article_publication_v2}")
for name in publication_refinement_v1 coverage_observables_v2; do
    if [[ -d "$FOPT_RESULTS_ROOT/$name" ]]; then arguments+=(--observables "$FOPT_RESULTS_ROOT/$name"); fi
done
if [[ -d "$FOPT_RESULTS_ROOT/lisa_reach_v2" ]]; then arguments+=(--reach "$FOPT_RESULTS_ROOT/lisa_reach_v2"); fi
if [[ "${FOPT_REPLACE_FIGURES:-0}" == 1 ]]; then arguments+=(--replace-figures); fi
"$python_executable" -m Articles.figures.publication "${arguments[@]}"
