#!/usr/bin/env bash
#SBATCH --job-name=gol-prep
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/prep-%j.log
# Build the pilot corpus: FineWeb-Edu sample-10BT shard 000 -> train,
# shard 001 -> first 10k docs valid, next 100k docs warmup; GPT-2 tokenizer.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source "${VENV:-/scratch/user/muhangao_tamu.edu/venvs/gol}/bin/activate"
ROOT="${ROOT:-/scratch/project/prj-02-tg-llms-and-vlms/muhan/gol}"
python scripts/fineweb_to_jsonl.py \
  --train-shard "$ROOT/raw/sample/10BT/000_00000.parquet" \
  --heldout-shard "$ROOT/raw/sample/10BT/001_00000.parquet" \
  --out "$ROOT/jsonl"
python -m gol.data --warmup "$ROOT/jsonl/warmup.jsonl" --train "$ROOT/jsonl/train.jsonl" \
  --valid "$ROOT/jsonl/valid.jsonl" --tokenizer openai-community/gpt2 \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --eos-token '<|endoftext|>' \
  --out "$ROOT/data/fineweb_edu_gpt2"
