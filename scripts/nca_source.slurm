#!/usr/bin/env bash
#SBATCH --job-name=nca-source-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log
# Template: set account/partition/QoS and activate the separately pinned env.
# One Python process; native DataParallel. DO NOT launch with torchrun.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the GOL repository root}"
: "${NCA_UPSTREAM:?Run repro.nca.bootstrap and set its checkout path}"
: "${NCA_RUN_ROOT:?Set a new output root, outside the upstream checkout}"
: "${NCA_ACK_RELEASE_DIFFERENCES:?Read docs/NCA_REPRODUCTION.md and set this to 1}"
[[ "$NCA_ACK_RELEASE_DIFFERENCES" == 1 ]]
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
python -m repro.nca.replay --stage source --upstream "$NCA_UPSTREAM" \
    --out "$NCA_RUN_ROOT/source_seed${SOURCE_SEED:-0}" --seed "${SOURCE_SEED:-0}" \
    --device cuda --accept-release-differences --execute
