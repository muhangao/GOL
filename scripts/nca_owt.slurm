#!/usr/bin/env bash
#SBATCH --job-name=nca-owt-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.log
# Template, not a runtime estimate. Native one-process DataParallel, not DDP.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the GOL repository root}"
: "${NCA_UPSTREAM:?Set verified upstream checkout}"
: "${NCA_RUN_ROOT:?Set a new output root}"
: "${OWT_DATA:?Set registered OpenWebText train.bin/val.bin directory}"
: "${NCA_ACK_RELEASE_DIFFERENCES:?Read the protocol and set this to 1}"
[[ "$NCA_ACK_RELEASE_DIFFERENCES" == 1 ]]
STAGE="${STAGE:-owt}"
[[ "$STAGE" == owt || "$STAGE" == scratch ]]
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
extra=()
if [[ "$STAGE" == owt ]]; then
    extra+=(--source-run "$NCA_RUN_ROOT/source_seed${SOURCE_SEED:-0}")
fi
python -m repro.nca.replay --stage "$STAGE" --upstream "$NCA_UPSTREAM" \
    --data "$OWT_DATA" --out "$NCA_RUN_ROOT/${STAGE}_seed${TEXT_SEED:-5}" \
    --seed "${TEXT_SEED:-5}" --device cuda --accept-release-differences --execute "${extra[@]}"
