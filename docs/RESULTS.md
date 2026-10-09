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
