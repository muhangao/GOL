# Validation of the source-audit follow-up

Executed locally on Python 3.13.5, PyTorch 2.10.0+cpu and NumPy 2.3.5.
No CUDA device or cluster filesystem was available in this validation environment.

- `python -m pytest -q`: **74 passed, 1 skipped** in 42.40 seconds.
  The skipped test is the existing optional JSON-tokenizer test; `tokenizers`
  was not installed locally. CI installs that extra.
- Both two-rank CPU Gloo/DDP tests actually ran, including the new test that
  changes global/micro batch sizes at the source-to-text boundary.
- New tests cover packed target roles, causal baseline features, disjoint fitted
  and held-out examples, weighted loss aggregation, empty groups, shifted model
  targets, legacy checkpoint loading, unchanged checkpoint hashes, post-transfer
  checkpoint rejection, source-seed collisions, phase-specific optimizer/batch
  settings, budget rounding, and exact CPU pause/resume across the handoff.
- Additional before/after regression: ran all four arms on the same tiny corpus
  using the original source modules from commit `73e44c6` and this revision.
  The original modules were verified by Git blob SHA. With no warmup override,
  each arm's final weights were bitwise equal, and final validation losses and
  estimated FLOPs were exactly equal. This is a CPU regression test, not a new
  pilot result. The new auditor also loaded and scored the original-version
  GoL warmup checkpoint without changing it.
- Ran the baselines-only audit with the original pilot generator: 4,096 fitting
  and 4,096 held-out sequences, seed 99173. All-token losses were 0.659953
  (unigram), 0.593449 (position only), 0.561346 (previous same cell) and 0.542235
  (same cell plus visible left/up). The known-rule reference scored 0.189449;
  it uses a fixed mean-density initial predictor, not latent-density posterior
  integration. **No H200-trained checkpoint was evaluated in this local run.**
- `python -m compileall -q gol tests` and `bash -n` on both audit launchers passed.

The existing pilot configuration and original result CSV/figures are unchanged.
The two new optimization configurations are unvalidated hypotheses, not fixes
known to improve transfer. This follow-up has not submitted a cluster job, run
GPU source auditing, or run a new FineWeb training sweep. CI status must be read
from the actual workflow run, not inferred from this document.
