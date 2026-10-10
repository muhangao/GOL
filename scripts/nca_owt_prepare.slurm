#!/usr/bin/env bash
#SBATCH --job-name=nca-owt-prepare
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log
# Reconstruct OWT (labeled "reconstructed"): pinned HF revision, nanoGPT's split
# policy (validation fraction 0.0005, seed 2357), tiktoken GPT-2 + one EOT/doc.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the GOL repository root}"
source "${VENV:-/scratch/user/muhangao_tamu.edu/venvs/nca-replay}/bin/activate"
python -m repro.nca.data prepare --out "${OWT_DATA:?}" \
  --revision "${OWT_REVISION:-79d93d786212f7344586290adb811d4ae6a1762c}" \
  --validation-fraction 0.0005 --split-seed 2357 --num-proc 60
