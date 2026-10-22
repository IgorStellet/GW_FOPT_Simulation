#!/bin/bash -l
#SBATCH --job-name=fopt-lisa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --partition=milliways-ext
#SBATCH --array=0-11%3
#SBATCH --signal=B:USR1@1800
#SBATCH --output=Articles/results/logs/lisa-%A_%a.out
#SBATCH --error=Articles/results/logs/lisa-%A_%a.err
set -euo pipefail
export FOPT_FRONTIER_STAGE=lisa
source "${SLURM_SUBMIT_DIR:?}/Articles/cluster/frontier_job.sh"
