# Experimental protocol

## Question and limits

Does fixed-rule Conway Game of Life warmup produce a Transformer body that
reaches lower held-out text loss after a fixed downstream training budget than
text warmup or random initialization?

This is a hypothesis-replication implementation. No particular published paper's
architecture, corpus, serialization, boundary condition or reported effect is
claimed to be reproduced. It cannot reconstruct an unpublished experiment from
a verbal description. No downstream performance advantage is hardcoded.

## GoL source

The rule is B3/S23: a dead cell becomes live with exactly three live neighbors;
a live cell survives with two or three. The default boundary is toroidal, with
an explicit dead-boundary option. Each trajectory draws an initial Bernoulli
board using a density sampled uniformly from the configured interval. The pilot
uses 12-by-12 boards and six frames. Extinct and periodic trajectories are kept,
so source complexity may decline over time; this is a known pilot limitation,
not an implicit filtering choice.

Each trajectory is serialized row-major as:

```
BOS, frame_0_cells, FRAME, frame_1_cells, FRAME, ..., frame_last_cells, FRAME, EOS
```

The source vocabulary is `dead=0, alive=1, FRAME=2, BOS=3, EOS=4`. Whole
trajectories are packed until a sequence has at least S+1 tokens, then truncated.
Every sequence begins at a trajectory boundary. Training predicts positions
1 through S from positions 0 through S-1 under a strict causal mask. All targets
contribute to the loss, including random initial frames and delimiters. No future
board is exposed when predicting its earlier cells. The configuration requires
at least two full frames to fit in a context.

## What is controlled at the handoff?

The controlled comparison is `gol_reset` versus `text_reset`:

- Same initial body per seed, downstream architecture, tokenizer and text corpus.
- Identical newly initialized text embedding/head per seed; tied and untied
  modes both supported. No trained source I/O values survive.
- All body parameters, including normalization, retained unchanged.
- AdamW state reset; a common downstream LR schedule starts from its beginning.
- Identical global downstream batches and validation windows.

The hashes in `transfer.json` check weight preservation, not scientific quality.
The reset controls measure body transfer, not a best-practice text-training
recipe. `text_continue` keeps its text I/O, AdamW moments and continuous schedule
across the corpus boundary. Its first-stage schedule consequently need not match
`text_reset`. This intentional practical reference must not be mistaken for a
single-variable optimizer ablation. The scratch arm has no warmup cost and fewer
total updates. No claim of superiority should rely only on beating a deliberately
reset text baseline.

## Compute estimate

For L layers, width d, SwiGLU intermediate width f, sequence length S and active
output vocabulary V, the estimate per prediction token is:

```
6 * [L * (4*d*d + 3*d*f) + d*V] + 12*L*S*d
```

The factor six is forward plus backward dense linear matmuls. The last term
counts dense attention QK and AV forward/backward. Tying weights does not remove
the head matmul. Embedding lookups, normalization, activation functions, softmax,
loss, AdamW updates, data generation/preparation, checkpointing, validation and
communication are excluded. Actual causal attention kernels can skip triangular
work. This is deliberately labeled an estimate, not exact or measured FLOPs.

A text-reference warmup budget is converted to a whole number of GoL updates by
ceiling division. Report the resulting overshoot and actual exposure counts;
this discretization is especially noticeable in the tiny smoke configuration.
For end-to-end resource claims, additionally account for excluded work and
hardware timing. The logged update seconds include data sampling, computation
and communication but exclude validation and checkpoint writing.

## Reading the curves

The primary observable is held-out cross-entropy on the same text tokenization.
Look at both downstream-token and total-compute axes. A lower starting loss,
larger absolute loss drop and better compute-to-target-loss are different
properties. A steeper slope from a worse initial loss alone does not establish
better learnability. Starting above a text control and later overtaking it is
interesting but still not a causal mechanism identification.

Use independent model seeds, predeclare budgets and examine robustness before
interpreting small differences. Do not select the best seed or best stopping
point on the validation set and then report it as an unbiased effect. A separate
held-out test corpus and larger runs are appropriate after choosing the setup.
The first PR intentionally stops before those expanded experiments.
