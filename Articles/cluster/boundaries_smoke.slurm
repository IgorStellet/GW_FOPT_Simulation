#!/bin/bash -l
#SBATCH --job-name=fopt-edge-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=milliways-ext
#SBATCH --array=0-2%3
#SBATCH --signal=B:USR1@900
#SBATCH --output=Articles/results/logs/boundaries-test-%A_%a.out
#SBATCH --error=Articles/results/logs/boundaries-test-%A_%a.err
export FOPT_STAGE=boundaries FOPT_PILOT=1
source "${SLURM_SUBMIT_DIR:?}/Articles/cluster/boundary_job.sh"
