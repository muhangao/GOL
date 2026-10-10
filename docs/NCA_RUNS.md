# NCA replay: executed runs (cluster log)

Released-code profile, upstream `bdd1c71`, 4x H200 per stage, native DataParallel, env `venvs/nca-replay`
(Python 3.11, torch 2.8.0+cu128, JAX 0.6.2 CPU backend as in the authors' requirements).
Run root: `/scratch/project/prj-02-tg-llms-and-vlms/muhan/gol/nca_runs`.

## OWT corpus (reconstructed)

`scripts/nca_owt_prepare.slurm`: `Skylion007/openwebtext` revision `79d93d786212f7344586290adb811d4ae6a1762c`,
validation fraction 0.0005, split seed 2357 (nanoGPT's split policy; the authors' split is unreleased, D11).
Result: train 9,035,582,489 tokens / 8,009,762 docs; val 4,434,606 tokens / 4,007 docs. Origin `reconstructed`.

## Seed pairing

The paper reports four seeds without their pairing. Pair 0 uses the released launchers' defaults; pairs 1-3 are
a recorded reconstruction choice. Each scratch run uses the same text seed as its NCA transfer run.

| Pair | NCA source seed | Text seed (NCA->OWT and scratch) | Note |
| --- | ---: | ---: | --- |
| 0 | 0 | 5 | Released launcher defaults |
| 1 | 1 | 6 | Reconstruction choice |
| 2 | 2 | 7 | Reconstruction choice |
| 3 | 3 | 8 | Reconstruction choice |

## Pair 0 source stage (completed)

5,000 successful updates, 163,840,000 input tokens, 157,920,000 supervised targets, `model_10.pth`,
source validation loss 6.2140, about 70 min including rule generation (18,000 rules, gzip threshold 0.50).
Text model: 1,816,565,760 parameters. OWT stage: 551,488 microbatches = ~17,234 updates; the native loop
validates and writes two 21.8 GB checkpoints every 100 updates (about 15 min per 100 updates, ~43 h per run).
