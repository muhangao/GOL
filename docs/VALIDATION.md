# Validation of the initial implementation

## Actually executed locally

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5; no CUDA device.

- `python -m compileall -q gol scripts`.
- Editable package install using existing dependencies:
  `pip install -e . --no-deps --no-build-isolation`.
- Toy-text preparation and a complete four-arm CPU smoke sweep, including both
  loss figures and the final-loss CSV.
- `python -m pytest -q`: 33 passed, 1 skipped on the initial test suite.
  The skipped test needs the optional `tokenizers` dependency, which was not
  installed in the local offline environment.

Tests cover Conway still life, blinkers, gliders, boundary semantics, trajectory
serialization, causal attention, BF16 rotary arithmetic on CPU, tied/untied I/O,
vocabulary-independent body initialization, exact body transfer, paired fresh
text I/O, budget rounding, text deduplication, corruption detection, finite
end-to-end losses, optimizer reset/continuation and exact CPU resumption within
warmup, at its boundary and within downstream training.

The two-rank CPU Gloo/DDP test actually executes a staged GoL run via `torchrun`,
including the interface swap. It compares against a single-process run with the
same global batch and deliberately uses an uneven validation partition. Final
weights/loss agree within the test tolerances. It is not a CUDA/NCCL validation.

## Not established by these checks

- No full-scale text run or three-seed scientific result has been produced.
- No GoL-over-text advantage has been established.
- CUDA/NCCL multi-GPU execution and BF16 GPU training were not run locally.
- The Hub download path and optional JSON-tokenizer path were not run locally.
- There is no claim of exact reproduction of a published NCA/GoL paper.

The provided CI workflow installs the optional tokenizer dependency to exercise
its local-JSON test without downloading a model. A successful CI run should be
reported separately; the existence of the workflow is not evidence that it ran.
