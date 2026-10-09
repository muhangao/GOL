#!/usr/bin/env bash
# Sequential arms/seeds; each invocation may use multiple GPUs on one node.
set -euo pipefail
CONFIG="${CONFIG:-configs/pilot.json}"
DATA="${DATA:-data/text}"
OUT="${OUT:-runs/pilot}"
NPROC="${NPROC:-1}"
DEVICE="${DEVICE:-auto}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a arms <<< "${ARMS:-scratch text_reset gol_reset text_continue}"
launch=(python)
if (( NPROC > 1 )); then
    launch=(torchrun --standalone --nproc-per-node="$NPROC")
fi
for seed in "${seeds[@]}"; do
    for arm in "${arms[@]}"; do
        run="$OUT/${arm}_seed${seed}"
        extra=()
        if [[ "${RESUME:-0}" == 1 && -f "$run/last.pt" ]]; then
            extra+=(--resume)
        fi
        "${launch[@]}" -m gol.train --config "$CONFIG" --data "$DATA" \
            --out "$run" --arm "$arm" --seed "$seed" --device "$DEVICE" "${extra[@]}"
    done
done
python -m gol.plot --runs "$OUT" --out "$OUT/plots"
