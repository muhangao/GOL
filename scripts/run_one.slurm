#!/usr/bin/env bash
#SBATCH --job-name=gol-run
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.log
# One arm/seed on one GPU. Submit one job per run to execute the sweep in parallel:
#   sbatch --job-name=gol-gol_reset-s0 --export=ALL,ARM=gol_reset,SEED=0 scripts/run_one.slurm
# Resubmitting a run that already has last.pt resumes it. STOP_AFTER=N pauses after
# absolute update N (e.g. the warmup update count for a source-only screen).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source "${VENV:-/scratch/user/muhangao_tamu.edu/venvs/gol}/bin/activate"
CONFIG="${CONFIG:-configs/pilot.json}"
DATA="${DATA:-/scratch/project/prj-02-tg-llms-and-vlms/muhan/gol/data/fineweb_edu_gpt2}"
OUT="${OUT:-/scratch/project/prj-02-tg-llms-and-vlms/muhan/gol/runs/pilot}"
run="$OUT/${ARM:?}_seed${SEED:?}"
extra=()
[[ -f "$run/last.pt" ]] && extra+=(--resume)
[[ -n "${STOP_AFTER:-}" ]] && extra+=(--stop-after "$STOP_AFTER")
export OMP_NUM_THREADS=4
nvidia-smi -L
python -m gol.train --config "$CONFIG" --data "$DATA" --out "$run" \
  --arm "$ARM" --seed "$SEED" --device cuda "${extra[@]}"
