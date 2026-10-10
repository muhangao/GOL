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

## Verified GitHub Actions execution (2026-10-09)

The following completed runs were checked through the GitHub Actions API. The
NCA integration job's actual logs were also read; success is not inferred from
the presence of a workflow file.

- Code commit: `278461d9896d8462048f84b81e1e4e8d439296ea`.
- PR merge checkout used by the NCA integration job:
  `09bed16f059dd1b1142b77ac91c034116d3da837`.
- [Pinned NCA replay smoke, run 37964712389](https://github.com/muhangao/GOL/actions/runs/37964712389):
  **completed / success**. Job `113936058613` completed dependency installation,
  the 30 contract/safety tests, exact upstream checkout verification, execution
  of the source and OWT drivers, and provenance artifact upload.
- [CPU tests, run 37964712501](https://github.com/muhangao/GOL/actions/runs/37964712501):
  **completed / success**.

The NCA job used Python 3.11.17, PyTorch 2.8.0+cpu, Transformers 4.53.0,
JAX/JAXlib 0.6.2, Flax 0.11.2 and NumPy 2.2.6. These are the principal version
pins; this is not a claim that every transitive dependency matches the authors'
original environment.

The logs confirm that upstream commit
`bdd1c71e04eb2e5a2c91481faf278e95846b4431` was fetched and verified, and that
`python -m repro.nca.integration_smoke` actually completed all three stages:

| Smoke stage | Successful optimizer steps | Input-token exposures | Supervised targets |
| --- | ---: | ---: | ---: |
| Original NCA source driver | 4 | 608 | 568 |
| Original OWT driver with the source checkpoint | 2 | 304 | 304 |
| Original OWT driver from scratch | 2 | 304 | 304 |

The source used the tiny two-layer, width-32 smoke model. The OWT driver used
small generated integer-token files, **not an OpenWebText corpus**. All three
completion records explicitly report `profile: smoke` and
`exact_paper_reproduction: false`. These values verify that the generator,
training entrypoints, checkpoint selection, handoff and scratch path execute;
they are not scientific transfer results.

The job uploaded the
[nca-smoke-provenance artifact](https://github.com/muhangao/GOL/actions/runs/37964712389/artifacts/11631974926),
containing JSON/JSONL records rather than model weights. The upload completed;
this verification did not independently download and audit every artifact file.

One source-stage warning remains visible in the successful log: Python warns
that `os.fork()` after JAX initialization may deadlock. This small run finished,
but it does not establish that the same worker configuration is safe at scale.

## Limits

Successful CPU smoke execution does not validate the 50%+ filter distribution,
full-size CUDA/DataParallel/FP16 stability, OpenWebText download/preparation,
164M-to-9B training, a transfer advantage, or exact equivalence to the paper.
It also does not establish the NCA-versus-C4 comparison; the unresolved data and
paper/code differences remain documented in `NCA_REPRODUCTION.md`.

This verification updates documentation only. It does not modify the replay
implementation or the original GoL configurations/results, submit a full-scale
training job, or relabel the GoL experiments as NCA reproductions. The successful
workflow links above refer specifically to code commit `278461d`, not to future
commits or unobserved runs.
