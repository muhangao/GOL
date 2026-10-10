# NCA reproduction: replay the published implementation first

## Objective and honest status

The next reference is **NCA -> OpenWebText versus scratch -> OpenWebText** using
the original authors' generator, tokenizer, model, optimizer, scheduler and
training loops. It is not another 124M FineWeb GoL setting. Existing GoL code,
configurations and negative results are unchanged.

Target: *Training Language Models via Neural Cellular Automata*,
[arXiv:2603.10055v1](https://arxiv.org/abs/2603.10055v1), especially Figure 2(a),
sections 3-4 and Appendix B/Table 2. The immutable code reference is
[danihyunlee/nca-pre-pretraining@bdd1c71](https://github.com/danihyunlee/nca-pre-pretraining/tree/bdd1c71e04eb2e5a2c91481faf278e95846b4431).

**This release supplies an executable, source-pinned code replay. It does not
claim that a positive result, or exact equivalence to every paper setting, has
been established.** The paper and released code disagree in material places.
Those conflicts are recorded below, not silently reconciled. Every run states
`exact_paper_reproduction: false` until the missing details are actually resolved.
Changing that metadata alone would not resolve them.

The first comparison does not establish NCA > C4: the release lacks an exact C4
warmup launcher/sweep selection. No substitute C4 recipe or Dyck experiment is
invented here. Obtain those settings before reproducing that separate claim.

## What is actually executed

`python -m repro.nca.replay` imports `src/nca_ppt.py` or
`src/openwebtext_pt.py` from a checksum-verified snapshot. It does **not** call
`gol.train`, port the generator to PyTorch, or use `gol.model.Decoder`.

| Item | Released-code reference |
| --- | --- |
| NCA | Original JAX/Flax random network, periodic 12x12 grids, 10 states |
| Encoding | Original 2x2 patch tokenizer, 10,000 patches + grid delimiters |
| Source I/O allocation | 64,000 input and output coordinates |
| Model | Original Llama wrapper, 24 layers, 32 heads, width 2048, MLP width 8192 |
| Source optimization | Adam defaults, LR 1e-4, microbatch 16, accumulation 2, no clipping/decay |
| Selected source checkpoint | `interval_save/model_10.pth`, 5,000 updates, 163,840,000 input-token exposures |
| Source LR horizon | 50,000 updates, 5,000-update warmup; selection does NOT shorten it |
| Source supervision | 987 unmasked targets per 1,024-token example; first grid masked |
| OWT | Original uint16 GPT-2 binary loader, full ~9B-token corpus, one pass |
| OWT optimization | AdamW defaults, LR 5e-4, batch 16 x accumulation 32, decay 1e-4, clip 1.0 |
| Handoff | Original embedding/head reinitialization; strict body-copy check; fresh OWT optimizer |
| Parallelism | Original one-process DataParallel; NOT torchrun/DDP |

The source schedule comes from the released NCA launcher: 100 epochs of 500
updates each. The OWT launcher explicitly recommends checkpoint 10. That point
is 163.84M input tokens, but only **157.92M supervised targets**. The wrapper
stops *after the original interval save* and asserts both observed counts.
Running the original 100 epochs all the way through would spend 1.6384B source
tokens. Rebuilding a 164M-token cosine schedule would be a different experiment.

The default source seed is 0 and text seed is 5, matching the released launchers.
First execute one pipeline to verify the reference. Appendix B reports four
seeds but not their complete pairing; additional explicit seed choices must be
recorded as a reconstruction, and receive identical treatment in the baseline.
There is no automatic 12-job or four-seed submission.

## Setup: a separate environment and immutable checkout

Run commands from this repository's root. Keep the original GoL environment for
old checkpoints; this path uses the author's older dependency versions.

```bash
python3.11 -m venv /path/to/venvs/nca-replay
source /path/to/venvs/nca-replay/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r repro/nca/requirements.txt
python -m repro.nca.bootstrap --out /path/to/nca-upstream-bdd1c71
```

The bootstrap verifies the commit and hashes of all imported source files, the
launchers, dependency lists and license. It will not overwrite another checkout
or fetch a moving branch. A resumed bootstrap accepts only an already correct
checkout. The upstream MIT license remains in each run snapshot.

The lock includes the principal upstream runtime versions. The requirements
file is a focused subset, not a claim to reproduce all transitive dependencies
or GPU drivers; actual versions, CUDA devices and JAX backend are recorded.
`--allow-environment-drift` is an explicit non-locked experiment, never a default.
The CPU workflow uses the same Torch release's CPU build.

## Data: no FineWeb substitution and no invented original split

The released `src/datasets/preprocess.py` prepares a math **test** split. It does
not define the paper's OWT train/validation split or provide hashes. The OWT
loader expects **`train.bin`, `val.bin`, little-endian uint16, GPT-2 tokens**.
Do not pass the existing uint32 FineWeb-Edu files to this path.

For existing assets supplied by the authors, record their provenance in a JSON
object (source, dataset version/split description, and any supplied hashes):

```bash
python -m repro.nca.data register --data-dir /path/to/author-owt \
  --origin declared_author_assets --provenance-json /path/to/owt-provenance.json
```

The tool scans IDs, counts tokens and hashes every byte. "Declared author assets"
is a recorded origin, not a claim of independent authentication.

When those assets are unavailable, explicitly reconstruct OWT instead:

```bash
python -m repro.nca.data prepare --out /path/to/reconstructed-owt \
  --revision FULL_40_CHARACTER_HF_DATASET_COMMIT \
  --validation-fraction YOUR_PREDECLARED_FRACTION --split-seed YOUR_PREDECLARED_SEED
```

This loads `Skylion007/openwebtext`, uses tiktoken `encode_ordinary` and one EOT
per document, and writes only actual tokens. The revision and split choices are
required, recorded inputs; no undisclosed "author split" default is inserted.
Reconstructed data remains labeled in every downstream result. Full-scale
execution rejects a train corpus below 8B tokens; this is a coarse guard against
accidentally using a pilot, not verification that the exact author corpus was
recovered. A matched source-code replay on reconstructed data is useful, but is
not enough to declare the exact published numerical result irreproducible.

## Plan, smoke, then the reference run

Planning is safe on a login node and requires no GPU/dependency imports:

```bash
python -m repro.nca.replay --stage source
python -m repro.nca.replay --stage owt
```

Before allocating full-scale compute, execute the *actual upstream drivers* with
an explicitly tiny smoke configuration in a fresh output directory:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m repro.nca.integration_smoke \
  --upstream /path/to/nca-upstream-bdd1c71 --out /path/to/nca-smoke
```

This is an engineering check, not small-scale scientific evidence. It uses a permissive complexity band solely for the tiny smoke,
uses toy integer tokens for the OWT loader, and checks the
source run, selected checkpoint, real upstream handoff and scratch path.

On an allocated multi-GPU node, after reviewing the discrepancy ledger:

```bash
export NCA_UPSTREAM=/path/to/nca-upstream-bdd1c71
export NCA_RUN_ROOT=/path/to/new-nca-runs
export OWT_DATA=/path/to/registered-owt
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m repro.nca.replay --stage source --upstream "$NCA_UPSTREAM" \
  --out "$NCA_RUN_ROOT/source_seed0" --accept-release-differences --execute

python -m repro.nca.replay --stage owt --upstream "$NCA_UPSTREAM" \
  --source-run "$NCA_RUN_ROOT/source_seed0" --data "$OWT_DATA" \
  --out "$NCA_RUN_ROOT/owt_seed5" --accept-release-differences --execute

python -m repro.nca.replay --stage scratch --upstream "$NCA_UPSTREAM" \
  --data "$OWT_DATA" --out "$NCA_RUN_ROOT/scratch_seed5" \
  --accept-release-differences --execute

python -m repro.nca.summarize --runs "$NCA_RUN_ROOT" --out "$NCA_RUN_ROOT/summary"
```

`nca_source.slurm` and `nca_owt.slurm` are explicit-submission templates. Set
account/partition/QoS, review resource limits, activate the pinned environment,
and `mkdir -p logs` before submitting. They request four GPUs and use native
DataParallel. Do not use the GoL `run_sweep.sh`, `configs/pilot*.json`, or torchrun.
The time limits are placeholders, not runtime estimates. Source generation and
GPU capacity still require a real-cluster preflight.

Keep OWT and scratch runs on the same code, corpus, hardware and text seed. Only
start the transfer after `selected_source.json` exists. A GoL `warmup.pt`, an
unselected source checkpoint, a hash mismatch, or the wrong budget is rejected.
All output directories must be new. Native resume omits RNG/dataloader/scaler
state needed for exact replay; this wrapper deliberately refuses implicit
resume rather than pretending it is exact. Save sufficient wall time and disk
space for the full run. Only load trusted checkpoints: upstream uses pickle.

## Paper/code discrepancy ledger

These are observations from the pinned source, not claims that the authors'
private runs used every released default. `protocol.py` also emits them in JSON.

| ID | Observed discrepancy or unknown |
| --- | --- |
| D01 | Table 2 says source effective batch 16; the launcher uses 16 with accumulation 2. DP divides this batch across GPUs. |
| D02 | The recommended epoch-10 checkpoint is taken at the end of LR warmup in a 100-epoch schedule, not at the end of a 164M-token cosine schedule. |
| D03 | Paper uses temperature 1e-3 and uniform initial cells; the launcher uses 1e-4 and `init_state` a sampled shared categorical distribution. |
| D04 | The patch alphabet is 10,002 including delimiters; allocated model I/O is 64,000. |
| D05 | Paper mentions tying. The source builder never enables it; OWT's second I/O reset would break it even if initially enabled. |
| D06 | `NCADataset` masks the first grid before shifting and rebuilds labels from `seq`; later delimiter labels are not masked. |
| D07 | The source bf16 flag does not select autocast dtype; CUDA defaults to FP16. The OWT launcher explicitly uses FP16. |
| D08 | Microbatch losses are not divided before backward in either released trainer. Adam/clip behavior is preserved. |
| D09 | `epoch+1 % generate_rules == 0` has a precedence issue: with 1, rules refresh only at epoch 0. The release reuses that bank thereafter. |
| D10 | NCA padding includes channels; gzip serializes int32 JAX patch tokens with delimiters removed. Do not substitute raw-cell bytes or a spatial-only pad. |
| D11 | Exact OWT binaries/split, selected C4 hyperparameters and original sweep results are absent. The math-only preprocessing script cannot supply them. |
| D12 | The actual wrapper uses MLP width 4d, attention dropout 0.1 and a biased source head. Record actual parameter counts rather than substituting a generic "Llama 1.6B". |

Source locations: `scripts/prepretraining/nca_prepretraining.sh`,
`scripts/pretraining/owt_ft.sh`, `src/nca_ppt.py`, `src/openwebtext_pt.py`,
`utils/nca.py`, `utils/models.py`, `utils/training_args.py`, `utils/tokenizers.py`.
Their immutable hashes are in `upstream.lock.json`. Resolving D01-D12 requires
matching configs/logs or author clarification; changing all of them together to
"reasonable" values would create another new experiment, not strict replication.

## Disclosed engineering repairs and instrumentation

The original checkout is never edited. A fresh snapshot receives exactly three
checked OWT repairs, with old/new hashes in `provenance.json`:

- R01: remove an unavailable, unused `DownstreamLanguageModel` import. Only the
  Llama path is enabled; no GPT-2 implementation is substituted.
- R02: call `_freeze_unfreeze_modules` on the original model rather than its
  DataParallel wrapper. The invoked model method is unchanged.
- R03: exclude a last OWT block if it lacks its shifted target. No target is
  padded and no fabricated token is appended. This can remove at most one block.

No source-generator, source-loop or model-math patch is applied. The source's
rule-refresh precedence, channel padding, nonuniform initial states, FP16
behavior and gradient accumulation semantics remain unchanged.

Instrumentation delegates to original functions. It logs actual target counts,
successful optimizer updates (separate from native iteration labels/FP16 skips),
model shapes and tying, optimizer/schedule arguments, source rule-bank hashes
and first-sequence hashes. Each sampled filtered rule bank is saved as a NumPy
array. Strict checkpoint loading and before/after body hashes are assertions,
not parameter changes. Local telemetry replaces external W&B upload. Final OWT
validation/checkpoint saving is added only after training ends, avoiding a new
pre-training validation pass that would perturb the original RNG schedule.

Validation uses the native mean of batch means, clearly labeled; it is not
silently converted to token-weighted aggregation. CSV summaries refuse mixed
corpora, hyperparameters, versions, wrappers, profiles or duplicate arm/seeds.
They report input-token budgets, not measured hardware FLOPs.

## Decision after the replay

First establish whether the pinned NCA-to-OWT path beats a matched scratch run.
Then confirm across the additional recorded seeds. Preserve negative curves as
well as positive ones. Do not optimize only the NCA arm on the validation set.
A full reproduction of NCA > C4 and all paper figures additionally needs the
missing C4 recipe and remaining original data/sweep details.

If this reference transfers, change **one** element to fixed Conway GoL and ask
which difference matters. If it does not, share the lock, resolved arguments,
corpus manifest, discrepancy ledger and curves with the authors before treating
it as a contradiction of the paper. No rescue of the existing GoL result is
assumed by this implementation.
