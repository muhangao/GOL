# GOL: Does Game of Life warmup improve subsequent text learning?

A loss-first experimental baseline. Train a small causal Transformer on Conway's
Game of Life (GoL), replace its token embedding and language-model head, then
continue on text. Compare held-out text loss with appropriate text controls.

**Scope:** this implements the experimental question, not an exact replication
of a particular paper, an unpublished project, or neural cellular automata (NCA).
It uses a fixed Conway B3/S23 rule, not a learned/random NCA rule family.
No improvement from GoL is assumed or established by this repository's tests.

## Quick start: an offline CPU smoke test

Python 3.10+ is required. Install the appropriate PyTorch build for your system.
The core training path needs only PyTorch and NumPy; no model weights are downloaded.

```bash
pip install -e '.[test,plot]'
python scripts/make_smoke_data.py --out data/smoke_raw
python -m gol.data \
  --warmup data/smoke_raw/warmup.txt \
  --train data/smoke_raw/train.txt \
  --valid data/smoke_raw/valid.txt \
  --tokenizer byte --out data/smoke
CONFIG=configs/smoke.json DATA=data/smoke OUT=runs/smoke \
  SEEDS=0 DEVICE=cpu bash scripts/run_sweep.sh
python -m pytest -q
```

The smoke corpus is generated toy prose, with a UTF-8 byte tokenizer and a tiny
model. It is for checking plumbing, **not** for making language-learning claims.
Prepared data and run directories are never silently overwritten.

## Four arms

| Arm | First stage | Transition | Common text stage |
| --- | --- | --- | --- |
| `scratch` | None | Fresh text I/O; random body | D updates |
| `text_reset` | Text warmup | Keep body; reset embedding/head, AdamW and LR schedule | D updates |
| `gol_reset` | GoL warmup | Keep body; replace embedding/head, reset AdamW and LR schedule | D updates |
| `text_continue` | Text warmup | Keep everything, including AdamW and the continuous LR schedule | D updates |

`gol_reset` vs `text_reset` is the controlled **body-transfer** comparison.
`text_continue` is the practical, uninterrupted text-only reference: it does not
pay an artificial interface-reset penalty. Its continuous LR schedule is
intentionally different from the reset controls. `scratch` diagnoses training
from a random body, but its final point has a smaller total compute budget.

Within a model seed, every arm starts with an identical Transformer body,
regardless of vocabulary size. `scratch`, `text_reset` and `gol_reset` receive
bitwise-identical fresh text embedding/head values at the start of their common
text stage. The transfer includes all Transformer blocks and the final RMSNorm.
Tied embeddings are the default, so embedding and head then share one parameter.
The tokenizer is a preprocessing interface, not a trainable part of the model.

All arms see the same downstream token windows in the same order. Model seeds
vary initialization, not the data schedule. Validation windows are fixed and
shared. Dropout is disabled. The first text validation measurement is taken
**before** any downstream update.

## Prepare actual text data

Provide separate local warmup, downstream-training and validation files. TXT
means one document per nonempty line; JSONL means one object with a string `text`
field per line. Multi-line documents should be stored as JSONL strings.

```bash
pip install -e '.[tokenize,plot,test]'
python -m gol.data \
  --warmup /path/to/warmup.jsonl \
  --train /path/to/train.jsonl \
  --valid /path/to/valid.jsonl \
  --tokenizer openai-community/gpt2 --eos-token '<|endoftext|>' \
  --out data/text
```

A local `tokenizer.json` path or directory can replace the Hub name. For a Hub
source, pin `--revision` to a commit for immutable provenance. The resolved
vocabulary, EOS ID, effective tokenizer JSON and its SHA256 are recorded. Only
the tokenizer is downloaded, not pretrained model weights or remote code.
Encoding disables implicit special-token insertion and explicitly appends one
EOS per document. All text arms use this same tokenized corpus.

Preparation streams documents into little-endian uint32 files, uses an on-disk
SQLite index for exact document deduplication across all splits, and writes a
manifest with token counts and checksums. Validation takes priority, followed by
warmup, then training. Duplicate removal counts are reported; inspect them.
This is **not** semantic or near-duplicate decontamination. Source splits must
still be chosen carefully. Preparation refuses to reuse an existing output
folder. `--verify-data` on training recomputes the full token-file checksums;
normal loading checks file sizes and sampled token IDs.

## Run a pilot

`configs/pilot.json` describes a 12-layer, width-768 decoder with 12 attention
heads, SwiGLU, RMSNorm and RoPE. With the GPT-2 vocabulary and tied I/O it has
about 124M text-stage parameters; the vocabulary-independent body is about 85M.
These are pilot choices, not claimed paper hyperparameters or tuned optima.

```bash
# One run; automatically chooses CUDA when available.
python -m gol.train --config configs/pilot.json --data data/text \
  --arm gol_reset --seed 0 --out runs/pilot/gol_reset_seed0

# Three model seeds and all four arms, sequentially; eight GPUs per run.
NPROC=8 CONFIG=configs/pilot.json DATA=data/text OUT=runs/pilot \
  bash scripts/run_sweep.sh

# Restart interrupted runs using their latest complete optimizer updates.
RESUME=1 NPROC=8 CONFIG=configs/pilot.json DATA=data/text OUT=runs/pilot \
  bash scripts/run_sweep.sh
```

The global batch is 128 sequences of 1,024 prediction targets. The downstream
stage has 2,048 updates (268,435,456 token exposures). The reference text warmup
has 128 updates (16,777,216 token exposures); GoL's update count depends on the
budget mode. Sampling is with replacement from each prepared split: token
exposures are not a count of unique corpus tokens. Check whether your corpus and
budget are adequate before increasing model scale.

`global_batch_size` must be divisible by `micro_batch_size * WORLD_SIZE`.
Gradient accumulation preserves the global batch across compatible world sizes.
The trainer supports `torchrun`, including the stage transition; the sweep
script is a single-node launcher. A cluster-specific starting template is in
`scripts/pilot.slurm`; set account/partition, environment and resources before
submitting it. No cluster job is submitted by this repository automatically.

## Budget matching: tokens are not FLOPs

A five-symbol GoL head is cheaper than a large text-vocabulary head. The two
budgets therefore cannot be claimed identical merely because sequence length
and update count match.

- `budget_match: "tokens"`: identical warmup prediction-token exposures; compute
  differs and is recorded.
- `budget_match: "estimated_flops"` (default): choose enough GoL updates to match
  the text reference's analytical training-matmul budget. Rounding can overshoot
  by less than one GoL update; token counts differ and are recorded.

The estimator explicitly includes the output projection. It is not a profiler,
wall-clock match or exact causal-kernel count. See [the protocol](docs/PROTOCOL.md)
for the formula and exclusions. Use the total-compute plot for efficiency claims;
do not compare unequal-cost final points and call them compute-matched.

## Outputs and resumption

Each run writes `metadata.json`, `initialization.json`, `metrics.jsonl`,
`last.pt`, and, after warmup, `warmup.pt` and `transfer.json`. Completed runs also
write `complete.json`. Metrics include text validation loss in nats/token,
phase steps, warmup/downstream tokens, total estimated training FLOPs and update
wall time. GoL warmup loss is not compared with text loss: text loss is undefined
for its five-symbol interface before transfer.

Checkpoints include optimizer state and the completed-update cursor. Data
sampling is stateless. Resumption rejects changed config, corpus manifest,
source code, seed, arm, world size or recorded runtime. It trims log entries
beyond the committed checkpoint. `--stop-after N` pauses at the absolute Nth
optimizer update and is useful for testing recovery. Only load trusted run
checkpoints. CPU bitwise resumption is tested; CUDA kernel nondeterminism can
prevent bitwise equivalence. Weights/optimizer files can be large and are not
committed to Git.

```bash
python -m gol.plot --runs runs/pilot --out plots/pilot
```

The plotter produces separate `loss_vs_text_tokens.png` and
`loss_vs_total_flops.png` files, plus `final_losses.csv`. It shows seed means and
one sample standard deviation (not confidence intervals), uses only common
observed checkpoints within an arm, and never extrapolates. The compute plot
also includes the uninterrupted text baseline's warmup measurements. Incomplete
runs are excluded unless `--include-incomplete` is specified. Different corpus,
config, code or runtime signatures are not silently aggregated.

## Current validation and limitations

See [validation notes](docs/VALIDATION.md) for the exact tests actually run.
A toy loss curve does not establish a GoL advantage, a causal initialization
mechanism, or a general measure of learnability. This first implementation has
no probes, router, looped model, NCA generator, or benchmark suite. Start with
held-out loss curves and reproducible controls before adding those components.
