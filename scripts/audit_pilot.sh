#!/usr/bin/env bash
# Read existing source checkpoints; never train or resume them.
set -euo pipefail
RUNS="${RUNS:-/scratch/project/prj-02-tg-llms-and-vlms/muhan/gol/runs/pilot}"
OUT="${OUT:-$RUNS/source_audit_v1}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
checkpoints=()
for seed in "${seeds[@]}"; do
    checkpoint="$RUNS/gol_reset_seed${seed}/warmup.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing source checkpoint: $checkpoint" >&2; exit 1; }
    checkpoints+=("$checkpoint")
done
(( ${#checkpoints[@]} > 0 )) || { echo "SEEDS must not be empty" >&2; exit 1; }
python -m gol.audit --checkpoints "${checkpoints[@]}" --out "$OUT" \
    --fit-sequences "${FIT_SEQUENCES:-4096}" --eval-sequences "${EVAL_SEQUENCES:-4096}" \
    --batch-size "${BATCH_SIZE:-8}" --seed "${AUDIT_SEED:-99173}" \
    --device "${DEVICE:-auto}" --dtype "${DTYPE:-auto}"
