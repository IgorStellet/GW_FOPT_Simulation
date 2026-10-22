#!/bin/bash -l
#SBATCH --job-name=fopt-edge-figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=milliways-ext
#SBATCH --output=Articles/results/logs/frontier-figures-%j.out
#SBATCH --error=Articles/results/logs/frontier-figures-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${FOPT_PYTHON_MODULE:-}" ]]; then module load "$FOPT_PYTHON_MODULE"; fi
: "${FOPT_RESULTS_ROOT:?Exporte FOPT_RESULTS_ROOT.}"
python_executable="${FOPT_PYTHON:-$SLURM_SUBMIT_DIR/.venv-che311/bin/python}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$SLURM_SUBMIT_DIR/Articles/cluster/job_environment.sh"
prepare_local_work "fopt-frontier-figures-$SLURM_JOB_ID"
export TMPDIR="$local_work"
arguments=(render --boundaries "${FOPT_BOUNDARIES_ROOT:-$FOPT_RESULTS_ROOT/boundaries_v1}"
    --output "${FOPT_FRONTIER_FIGURES_ROOT:-$FOPT_RESULTS_ROOT/article_frontiers_v1}"
    --C-limits "${FOPT_FIGURE_C_MIN:-0}" "${FOPT_FIGURE_C_MAX:-4.1}"
    --observation-years "${FOPT_OBSERVATION_YEARS:-4}" --snr-threshold "${FOPT_SNR_THRESHOLD:-10}" --bundle)
edge_data="${FOPT_EDGE_DATA_ROOT:-$FOPT_RESULTS_ROOT/edge_observables_v1}"
lisa_data="${FOPT_LISA_DATA_ROOT:-$FOPT_RESULTS_ROOT/lisa_reach_v1}"
if [[ -d "$edge_data" ]]; then arguments+=(--observables "$edge_data"); fi
if [[ -d "$lisa_data" ]]; then arguments+=(--reach "$lisa_data"); fi
"$python_executable" -m Articles.figures.frontiers "${arguments[@]}"
