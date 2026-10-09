# NCA replay validation

## Locally executed before submission

Environment: Python 3.13.5, PyTorch 2.10.0+cpu. No CUDA device, network access,
Flax or Transformers was available in the working container.

- `python -m pytest tests/test_nca_replay.py -q`: 30 passed.
- `python -m compileall -q repro tests`: passed.
- `bash -n scripts/nca_source.slurm scripts/nca_owt.slurm`: passed.
- Printed source/OWT plans and checked the 163,840,000 input-token versus
  157,920,000 supervised-target accounting and the unchanged 50,000-step LR horizon.

These are contract and safety tests, not an execution of the NCA model. They
cover budget arithmetic, explicit smoke labeling, reference architecture flags,
exact-match repairs, source hashes, corpus type/size/hash checks, checkpoint
receipt validation and immutable body hashing.

## Actual upstream integration

The separate `Pinned NCA replay smoke` GitHub Actions workflow installs the
principal pinned author dependencies, fetches/verifies commit bdd1c71, and
executes the original source driver, original OWT handoff and scratch driver at
tiny CPU sizes. Its real status/logs, not this file's existence, determine whether
that integration passed. It uploads JSON/JSONL provenance, not model weights.

Even successful smoke execution would not validate the 50%+ filter distribution,
full-size CUDA/DataParallel/FP16 stability, OpenWebText download/preparation,
164M-to-9B training, a transfer advantage, or exact equivalence to the paper.
No full-scale job was submitted by this PR update. The existing GoL results are
unchanged and are not relabeled as NCA reproduction results.
