# Pilot result: FineWeb-Edu, 124M decoder, 3 seeds

Run on 2026-10-08, git `d3c0ec8` on branch `codex/gol-warmup-reproduction`, `configs/pilot.json` unchanged.

## Setup actually executed

- Corpus: FineWeb-Edu `sample/10BT`. Shard `000_00000` is the downstream train split (750.0M tokens, 723,812 docs).
  Shard `001_00000`: first 10,000 docs are valid (11.0M tokens), next 100,000 are text warmup (101.7M tokens).
  GPT-2 tokenizer, revision `607a30d7`. Exact dedup removed 20 warmup and 2,188 train docs.
  Built by `scripts/prepare_fineweb.slurm`.
- One H200 per run (world size 1, 32-step gradient accumulation), BF16, torch 2.13.0+cu130.
  All 12 runs (4 arms x seeds 0/1/2) ran concurrently via `scripts/run_one.slurm`; about 31-34 min each.
- Budget: 128 text-warmup updates (16.8M tokens) vs. 176 GoL-warmup updates (23.1M tokens), matched by estimated
  FLOPs (2.437e17 total for both). Common text stage is 2,048 updates (268M tokens). Scratch uses 2.294e17 FLOPs.
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
`loss_vs_total_flops.png` in the same folder.

## Reading

- **GoL warmup hurt in this setup.** `gol_reset` ends 0.20 nats above the controlled `text_reset` baseline
  and 0.22 above `scratch`, consistently across all three seeds (worst text arm 3.737 vs. best GoL 3.906).
  The gap narrows over training (0.43 at step 256, 0.20 at step 2048) but never closes within 2,048 updates.
- **The handoff is not trivially broken.** GoL training loss fell to ~0.53 nats/token, and `transfer.json`
  confirms bitwise-identical body weights across the I/O swap. The deficit appears to come from the body learned on
  GoL, not from a failed transfer.
- **Resetting I/O and AdamW costs something by itself.** `text_reset` finishes 0.023 nats behind `scratch`
  even though it used 6% more compute. `text_continue` is best (3.689), but only 0.022 below scratch after spending the
  same extra FLOPs on text warmup.
- On the total-compute axis, `scratch` and `text_continue` coincide; neither reset arm catches up.

## Caveats

Single small configuration: 124M model, 12x12 toroidal boards with 6 frames, ~8% of the downstream budget spent on
warmup, untuned LR. Other choices may behave differently, e.g. a learned/random NCA rule family, longer
or larger GoL warmup, a lower LR or no AdamW reset after transfer, or partial body freezing. These results support
"no benefit and a measurable cost" only for this protocol. Validation loss was used for monitoring only; nothing
was selected on it.
