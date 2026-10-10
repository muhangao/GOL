#!/usr/bin/env bash
#SBATCH --job-name=gol-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=gol-%j.log

# Template only: set partition/account, activate your environment and adjust
# resources/time to your cluster. This file does not submit any job by itself.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=4
export NPROC="${NPROC:-8}"
export DEVICE=cuda
bash scripts/run_sweep.sh
