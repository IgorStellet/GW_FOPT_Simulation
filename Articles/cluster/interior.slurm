#!/bin/bash -l
#SBATCH --job-name=fopt-interior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --partition=milliways-ext
#SBATCH --array=0-2%3
#SBATCH --signal=B:USR1@900
#SBATCH --output=Articles/results/logs/interior-%A_%a.out
#SBATCH --error=Articles/results/logs/interior-%A_%a.err
export FOPT_STAGE=interior FOPT_PILOT=0
source "${SLURM_SUBMIT_DIR:?}/Articles/cluster/boundary_job.sh"
