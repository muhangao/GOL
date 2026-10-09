#!/usr/bin/env bash
#SBATCH --job-name=gol-source-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log
# Create logs/ before sbatch; supply your cluster's account/partition/QoS if needed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source "${VENV:-/scratch/user/muhangao_tamu.edu/venvs/gol}/bin/activate"
export OMP_NUM_THREADS=4
export DEVICE=cuda
bash scripts/audit_pilot.sh
