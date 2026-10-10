# Pilot result: FineWeb-Edu, 124M decoder, 3 seeds

Run on 2026-10-08, git `a0f3b0a` on branch `codex/gol-warmup-reproduction`, `configs/pilot.json` unchanged.

## Setup actually executed

- Corpus: FineWeb-Edu `sample/10BT`. Shard `000_00000` is the downstream train split (750.0M tokens, 723,812 docs).
  Shard `001_00000`: first 10,000 docs are valid (11.0M tokens), next 100,000 are text warmup (101.7M tokens).
  GPT-2 tokenizer, revision `607a30d7`. Exact dedup removed 20 warmup and 2,188 train docs.
  Built by `scripts/prepare_fineweb.slurm`.
- One H200 per run (world size 1, 32-step gradient accumulation), BF16, torch 2.13.0+cu130.
  All 12 runs (4 arms x seeds 0/1/2) ran concurrently via `scripts/run_one.slurm`; about 31-34 min each.
- Budget: 128 text-warmup updates (16.8M tokens) vs. 176 GoL-warmup updates (23.1M tokens), matched by estimated
  FLOPs (2.437e17 total for both, up to whole-update rounding). Common text stage is 2,048 updates (268M tokens).
  Scratch uses 2.294e17 FLOPs.
- Run directories: `/scratch/project/prj-02-tg-llms-and-vlms/muhan/gol/runs/pilot`.

## Held-out text loss (nats/token, mean ± sample SD over 3 seeds)

| Text step | scratch | text_reset | gol_reset | text_continue |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.963 ± 0.012 | 10.959 ± 0.016 | 10.930 ± 0.012 | 6.162 ± 0.047 |
| 64 | 6.837 ± 0.085 | 6.769 ± 0.119 | 7.033 ± 0.034 | 5.704 ± 0.054 |
| 256 | 5.378 ± 0.053 | 5.397 ± 0.051 | 5.801 ± 0.104 | 4.907 ± 0.035 |
| 512 | 4.592 ± 0.040 | 4.641 ± 0.030 | 5.043 ± 0.089 | 4.337 ± 0.014 |
| 1024 | 3.998 ± 0.010 | 4.027 ± 0.007 | 4.295 ± 0.042 | 3.931 ± 0.009 |
| 1536 | 3.791 ± 0.004 | 3.815 ± 0.005 | 4.029 ± 0.025 | 3.759 ± 0.008 |
| **2048** | **3.711 ± 0.003** | **3.734 ± 0.004** | **3.931 ± 0.021** | **3.689 ± 0.007** |

Per-seed final values: `results/pilot_fineweb/final_losses.csv`. Plots: `loss_vs_text_tokens.png`,
`loss_vs_total_flops.png` in the same folder. The original CSV, figures and configuration are preserved.

## Reading

- **GoL warmup hurt in this setup.** `gol_reset` ends 0.20 nats above the controlled `text_reset` baseline
  and 0.22 above `scratch`, consistently across all three seeds (worst text arm 3.737 vs. best GoL 3.906).
  The gap narrows over training (about 0.40 at step 256 from the table, 0.20 at step 2048), but never closes
  within 2,048 updates.
- **Copy integrity is verified; transfer compatibility is not.** `transfer.json` confirms bitwise-identical
  body weights across the I/O swap. This excludes accidental body changes during copying, not a mismatch between
  a source-trained body and randomly initialized text I/O. The result is negative transfer under this handoff,
  not a causal identification of why it happens.
- **A source training loss near 0.53 does not establish rule learning.** The aggregate includes random initial
  cells, evolution cells and separators. Cheap conditional-frequency predictors may perform well without
  implementing the full update rule. The trained checkpoints need a matched held-out, segmented source audit;
  see [SOURCE_AUDIT.md](SOURCE_AUDIT.md). No such checkpoint-audit result is inserted into this original table.
- **The reset protocol did not pay off.** `text_reset` finishes 0.023 nats behind `scratch` despite using about
  6% more estimated compute. `text_continue` is best (3.689), 0.022 below scratch after extra text training.
  These comparisons do not isolate I/O reset from optimizer/schedule changes; the continuous arm has a different
  LR schedule. The total-compute curves show scratch and continuous text approximately tracking each other.

## Caveats

Single small configuration: 124M model, fixed Conway B3/S23, five-symbol source vocabulary, 12x12 toroidal boards
and six frames, untuned LR. GoL warmup is about 8.6% of downstream token exposures but about 6.3% of downstream
estimated training FLOPs (about 5.9% of total FLOPs); these are different denominators and quantities.
This is not an exact NCA-paper reproduction. Different source rules, patch encodings, optimization settings or
handoffs require separate experiments and may or may not help. Validation loss was used for monitoring only;
nothing was selected on it. New diagnostics and configuration options do not invalidate or replace this negative result.

## Follow-up (2026-10-09): source audit, warmup screens, and a transfer re-test

Executed per [SOURCE_AUDIT.md](SOURCE_AUDIT.md) on single H200s. Audit: 4,096 fit + 4,096 held-out source
sequences, audit seed 99173, identical held-out indices for every checkpoint. Raw outputs are in
`results/source_audit/{orig_pilot,warmup_lr1e4,warmup_batch16}/`.
The screens and transfer runs used git `5ce7175`. Run directories are
`runs/screen_warmup_{lr1e4,batch16}` under the project path above.

### Held-out source loss (nats/token)

| Predictor | Source updates | all | initial cells | evolved cells | evolved alive | evolved dead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_cell_left_up` (best cheap baseline) | — | 0.542 | 0.671 | 0.498 | 1.156 | 0.271 |
| `known_rule_reference` (not a floor) | — | 0.189 | 0.674 | 0.000 | 0.000 | 0.000 |
| original pilot, seed 0 / 1 / 2 | 176 | 0.520 / 0.349 / 0.311 | 0.658 / 0.674 / 0.670 | 0.472 / 0.224 / 0.172 | 1.068 / 0.434 / 0.331 | 0.266 / 0.151 / 0.117 |
| `pilot_warmup_lr1e4`, seed 0 / 1 / 2 | 176 | 0.412 / 0.382 / 0.339 | 0.662 / 0.665 / 0.663 | 0.318 / 0.274 / 0.214 | 0.647 / 0.549 / 0.455 | 0.204 / 0.178 / 0.131 |
| `pilot_warmup_batch16`, seed 0 / 1 / 2 | 1,405 | 0.184 / 0.184 / 0.184 | 0.654 / 0.654 / 0.653 | 0.0001 / 0.0000 / 0.0000 | 0.0001 / 0.0001 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |

- The original 176-update source stage was **under-trained and seed-unstable**. Seed 0 barely beats the cheapest
  local-statistics baseline on evolved cells (0.472 vs 0.498); seeds 1-2 are clearly better but far from the
  deterministic-rule value of 0. Downstream text loss did not track this (3.940 / 3.946 / 3.906).
- Lower warmup LR (1e-4) is more consistent across seeds but still far from the rule.
- Source batch 16 (same token and estimated-FLOP budget, 8x the updates) drives held-out evolved-cell loss to
  ~1e-4 on every seed. Total loss is slightly below the fixed-density known-rule reference because the model also
  adapts its initial-cell prediction to the sampled board (0.653 vs 0.674). These checkpoints predict held-out
  B3/S23 transitions essentially perfectly; this does not identify *how* they compute them.

### Transfer with the batch-16 source stage

`gol_reset` resumed from the screened `warmup.pt` checkpoints; `text_reset` used the same warmup override
(1,024 text-warmup updates at batch 16, same reference budget). The downstream stage is unchanged. `scratch` and
`text_continue` are unaffected by the override, so the original pilot runs serve as their references (same corpus,
text batches, validation windows and model seeds; code differs only by the PR's warmup-override changes, which
the AUDIT_VALIDATION regression showed to be bitwise-neutral without an override).

| Text step | scratch (pilot) | text_continue (pilot) | text_reset (pilot) | gol_reset (pilot) | text_reset (b16) | **gol_reset (b16)** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10.963 ± 0.012 | 6.162 ± 0.047 | 10.959 ± 0.016 | 10.930 ± 0.012 | 11.005 ± 0.044 | 10.958 ± 0.021 |
| 256 | 5.378 ± 0.053 | 4.907 ± 0.035 | 5.397 ± 0.051 | 5.801 ± 0.104 | 5.104 ± 0.008 | 5.719 ± 0.090 |
| 1024 | 3.998 ± 0.010 | 3.931 ± 0.009 | 4.027 ± 0.007 | 4.295 ± 0.042 | 3.974 ± 0.011 | 4.377 ± 0.071 |
| **2048** | **3.711 ± 0.003** | **3.689 ± 0.007** | **3.734 ± 0.004** | **3.931 ± 0.021** | **3.717 ± 0.009** | **4.011 ± 0.051** |

Per-seed finals (b16): `gol_reset` 4.021 / 4.057 / 3.956; `text_reset` 3.726 / 3.716 / 3.708.
CSV and figures: `results/warmup_batch16_transfer/`.

- **Learning the rule did not rescue transfer; it made it worse.** With a source stage that predicts held-out GoL
  transitions essentially perfectly, `gol_reset` ends 0.29 nats behind the matched `text_reset` and 0.08 nats
  behind the under-trained original GoL body. Every GoL seed is worse than every text or scratch seed.
- The same warmup change helped the text control (3.734 to 3.717), so it is not a generically harmful optimizer
  setting.
- With two GoL source settings, better source prediction coincided with worse downstream text loss. This is two
  settings and three seeds each, not a dose-response measurement, and it does not identify the mechanism
  (e.g. body specialization, weight-norm growth, or I/O incompatibility).
- Not tested: lr1e4 transfer, interface-adaptation (e.g. training new text I/O with a frozen body first),
  partial body transfer, NCA/random-rule sources.
