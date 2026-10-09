# Audit source loss before launching another text-training sweep

The first pilot showed negative transfer under its specified protocol. Keep that
result. A source training loss near 0.53 is not, by itself, evidence of learning
Conway's transition rule. Likewise, identical body hashes verify copying, not
compatibility with a new embedding/head. This follow-up measures source loss;
it does not assume that a different setting will produce positive transfer.

## 1. Evaluate the existing three warmup checkpoints

From the repository root in the existing environment, submit one read-only audit:

```bash
mkdir -p logs
sbatch scripts/audit_source.slurm
```

Supply account/partition/QoS flags as required by your cluster. The template
reuses the environment path from the pilot launcher; override `VENV` as needed.
On an already allocated GPU, the equivalent is:

```bash
DEVICE=cuda bash scripts/audit_pilot.sh
```

The default `RUNS` is the pilot directory documented in `RESULTS.md`. Each
`gol_reset_seed{0,1,2}/warmup.pt` must exist. Missing checkpoints fail explicitly;
the launcher never silently substitutes a random model or baselines-only run.
`SEEDS`, `RUNS`, `OUT`, `FIT_SEQUENCES`, `EVAL_SEQUENCES`, `BATCH_SIZE`,
`AUDIT_SEED`, `DEVICE` and `DTYPE` may be overridden in the environment.

Default output: `$RUNS/source_audit_v1/report.json` and `losses.csv`. Existing
output directories are refused. Choose a different `OUT` for repeat evaluations.
The command does not update weights or optimizers, rewrite checkpoints, or
require the FineWeb corpus. It accepts legacy schema-1 source checkpoints even
though training code has changed. A post-transfer `last.pt` is not a source
checkpoint and is rejected. A paused, pre-transfer `last.pt` is acceptable.

Generic invocation:

```bash
python -m gol.audit \
  --checkpoints /path/to/gol_reset_seed0/warmup.pt \
                /path/to/gol_reset_seed1/warmup.pt \
                /path/to/gol_reset_seed2/warmup.pt \
  --fit-sequences 4096 --eval-sequences 4096 --batch-size 8 \
  --device cuda --out /path/to/new-audit-directory
```

Audit configuration comes from each checkpoint, not today's pilot defaults.
Checkpoints must agree on the generator and sequence length. The JSON records
checkpoint SHA256s, original model seeds/configs, source update counts, source
code provenance, audit/generator/loader hashes, runtime, and exact data indices.
Only load trusted local checkpoints. Loading uses `weights_only=True`, CPU
mapping and read-only mmap; optimizer tensors are not used or moved to the GPU.

## 2. What to compare

All predictors see the identical held-out sequences, with the same causal target
alignment as training. The report separates `all`, `initial_cells`,
`evolved_cells`, `delimiters`, `evolved_alive`, `evolved_dead`, and each frame's
cells. Each group includes token count, summed negative log likelihood and mean
loss in nats/token. Initial/evolved/delimiter sums reconstruct the total; empty
groups have a null mean, never a fabricated zero.

The pilot's packing produces **288 initial cells, 727 evolved cells and 9
separator targets per 1,024 targets**. A six-frame trajectory occupies 872
tokens, so packing appends another random initial board. Roles are derived from
the serialization grammar, including restarts and truncated final frames.

Four small fitted baselines are included:

- `unigram`: unconditional five-symbol frequency.
- `position_only`: conditional frequency at each serialized target position.
- `previous_same_cell`: conditions on whether the target is initial/evolved and,
  for evolved cells, its state in the previous frame.
- `same_cell_left_up`: additionally reads visible left/up cells of the current
  frame. These accesses never wrap into unseen right/bottom cells.

The latter two know separator positions from the grammar, but neither reads the
complete previous eight-neighbor neighborhood or implements the GoL rule. A
smoothing pseudocount (default 0.5) keeps unseen table entries finite. Tables are
fit on indices `[0, fit_sequences)` and scored on disjoint indices
`[fit_sequences, fit_sequences + eval_sequences)` under a new seed. Checkpoint
and baseline scores are all held-out; do not compare a baseline's held-out loss
with the old last training-batch loss as though those were matched evaluations.
The audit rejects a seed matching any checkpoint's source-training data seed.

`known_rule_reference` explicitly applies B3/S23 to the causally visible previous
frame and predicts separators from the grammar. For initial random cells it
uses the configured mean density, not the sampled hidden density. It is a
reference computation, **not a learned model, Bayes-optimal loss floor, or a
threshold that must be reached for transfer**. Its overall number differs from
a reference that integrates the latent-density posterior.

Focus first on evolved-cell loss and the alive/dead breakdown. Beating these
limited baselines is evidence of better source prediction, not a complete proof
of rule learning; matching them is not proof that no rules were learned. None
of these source losses alone establishes downstream transfer or its mechanism.

A CPU-only baselines diagnostic, without claiming checkpoint evaluation:

```bash
python -m gol.audit --config configs/pilot.json --device cpu \
  --fit-sequences 4096 --eval-sequences 4096 --batch-size 64 \
  --out /path/to/new-baselines-directory
```

## 3. Independent warmup optimization without silently changing budgets

The existing `configs/pilot.json` and original CSV/plots are unchanged. An
optional top-level `warmup` object now overrides source-stage `lr`,
`min_lr_ratio`, `lr_warmup_steps`, `global_batch_size`, `micro_batch_size`,
`weight_decay` and `grad_clip`. Omitted fields inherit `train`. Overrides apply
to BOTH `gol_reset` and `text_reset`; they do not change downstream training or
the uninterrupted `text_continue` reference. Scratch remains unchanged.

Two separate, unvalidated settings are supplied, not an all-at-once recipe:

- `configs/pilot_warmup_lr1e4.json`: only warmup peak LR becomes 1e-4.
- `configs/pilot_warmup_batch16.json`: only warmup global batch becomes 16
  (micro batch stays 4), increasing update count at the same reference budget.

The reference budget remains `train.warmup_reference_steps *
train.global_batch_size * seq_len`, optionally multiplied by text FLOPs/token.
Changing source batch recomputes the required update count. Whole-update
rounding may overshoot by less than one source update, in BOTH budget modes.
For GPT-2's 50,257-symbol vocabulary, the original GoL warmup has 176 updates;
with source batch 16 it has 1,405. The corresponding text-reset warmup has
1,024 updates. Report actual budgets, not just step counts.

Inspect the resolved plan without a GPU or a new run directory:

```bash
python -m gol.train --config configs/pilot_warmup_batch16.json \
  --data /path/to/prepared-text --arm gol_reset --plan-only
```

A source-only screening run can use `--stop-after` with the warmup update count
printed by that command. Pausing exactly at the boundary now writes `warmup.pt`
before any text I/O transfer. Do not use a GoL step count for the text-reset arm.
An unchanged run can later continue with `--resume`.

Use NEW output directories for altered settings. Training resumption still
rejects changed code/config/corpus/runtime. This PR does not disable that guard
or migrate old training states: evaluate old checkpoints with `gol.audit`; keep
the original code checkout for exactly resuming old pilot training. Existing
legacy metrics can still be plotted with `gol.plot` without loading checkpoints.

## Decision before more compute

Run the source audit first. If source prediction is barely above cheap local
statistics, screen source optimization before another full text sweep. If it is
clearly stronger, an interface-adaptation or targeted transfer control becomes
more informative. The NCA rule family, patch vocabulary, filtering and optimizer
recipe are separate experiments, not implemented or claimed reproduced here.
No cluster job is submitted by this PR.
