"""The released-code reference, not a silent reconstruction of missing settings.

No GPU imports here: plan, budget, hash and data checks can run on a login node.
The scientific values below follow the pinned *launchers and executable code*.
Paper/code disagreements remain in the plan rather than being guessed away.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "upstream.lock.json").read_text())

DISCREPANCIES = {
    "D01": "Paper Table 2 effective source batch 16; launcher 16 x accumulation 2 = 32 (DataParallel splits, not multiplies, this batch).",
    "D02": "The recommended epoch-10 source checkpoint has 163,840,000 input-token exposures; its scheduler still has a 100-epoch/50,000-update horizon and 5,000-update warmup. Do not shorten that horizon.",
    "D03": "Paper temperature 1e-3 and uniform initial cells; released launcher temperature 1e-4 and NCA.init_state samples a shared random categorical distribution.",
    "D04": "10,000 patch symbols plus two delimiters are used, but the released model allocates 64,000 input/output coordinates.",
    "D05": "Paper says weight tying; create_llama_model does not enable it, and downstream reinit_embeddings replaces both weights without retying. Replay keeps the released untied behavior.",
    "D06": "Released NCADataset masks the first grid (37 shifted target positions), not all-token prediction as the paper equation might suggest. It rebuilds targets from seq and supervises later delimiters.",
    "D07": "Source CLI says bf16, but its autocast calls omit dtype; CUDA therefore uses the PyTorch default FP16, with a new GradScaler each epoch. Replay records this instead of relabeling it BF16.",
    "D08": "Both released trainers sum microbatch gradients rather than dividing each loss by accumulation. This is preserved, including clipping/Adam consequences.",
    "D09": "The released expression epoch+1 % generate_rules == 0 refreshes rules only at epoch 0 when generate_rules=1. Replay preserves that precedence and the resulting rule-bank reuse.",
    "D10": "NCANetwork uses jnp.pad(x, pad_width=1, mode='wrap'), including the channel dimension. Gzip scores serialize JAX patch-token bytes, not one raw byte per cell. Neither is replaced by a new generator.",
    "D11": "The original OpenWebText split, bin hashes, exact C4 warmup configuration and sweep selections are not released. Reconstruction must be labeled; C4/Dyck claims remain outside this first replay.",
    "D12": "The released model has intermediate_size=4*hidden, attention_dropout=0.1 and an NCA output bias. Report actual parameter count rather than assuming its 1.6B label is exact."
}


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_checkout(root: Path, *, check_git: bool = True) -> dict:
    root = Path(root).resolve(strict=True)
    observed = {}
    for rel, expected in LOCK["files"].items():
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked upstream file: {path}")
        actual = blob_sha(path.read_bytes())
        if actual != expected:
            raise ValueError(f"Upstream content drift: {rel}: {actual} != {expected}")
        observed[rel] = actual
    if check_git:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if sha != LOCK["commit"]:
            raise ValueError(f"Wrong upstream HEAD: {sha}")
        # Reject tracked edits anywhere, not just the files that are imported.
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    return observed


def dependency_report() -> dict:
    result = {}
    for name, expected in LOCK["dependencies"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        result[name] = {"expected": expected, "actual": actual,
                        "matches": actual is not None and actual.split("+")[0] == expected}
    return result


def source_settings(smoke: bool = False) -> dict:
    values = dict(seed=0, grid=12, patch=2, num_colors=10, seq_len=1024,
                  batch_size=16, grad_accumulation_steps=2, learning_rate=1e-4,
                  num_epochs=100, warmup=10, n_layer=24, n_head=32, n_embd=2048,
                  temperature=1e-4, train_num_rules=16000, val_num_rules=2000,
                  train_num_sim=500, val_num_sim=100, dT=1, vocab_size=64000,
                  filter_rules_threshold=0.5, filter_rules_upper_bound=1.0,
                  filter_rules_mode="gzip", init_rollout_steps=10,
                  generate_rules=1, min_grid=1, val_freq=500,
                  log_grad_freq=100, mixed_precision="bf16")
    if smoke:
        values.update(grid=4, num_colors=2, seq_len=76, batch_size=2,
                      grad_accumulation_steps=1, num_epochs=2, warmup=0.5,
                      n_layer=2, n_head=2, n_embd=32, train_num_rules=4,
                      val_num_rules=2, train_num_sim=2, val_num_sim=1,
                      vocab_size=18, init_rollout_steps=1, val_freq=2,
                      mixed_precision="none", filter_rules_threshold=0.0,
                      filter_rules_upper_bound=10.0)
    return values


def source_budget(settings: dict, stop_epoch: int) -> dict:
    s = settings
    if not 1 <= stop_epoch <= s["num_epochs"]:
        raise ValueError("Invalid selected source epoch")
    batch = s["batch_size"] * s["grad_accumulation_steps"]
    per_epoch = s["train_num_sim"]
    grid_len = (s["grid"] // s["patch"]) ** 2 + 2
    ignored = s["min_grid"] * grid_len - 1  # shift happens AFTER masking
    if not 0 <= ignored < s["seq_len"]:
        raise ValueError("Invalid source supervision length")
    updates = stop_epoch * per_epoch
    return dict(selected_epoch=stop_epoch, effective_global_batch=batch,
                updates_per_epoch=per_epoch, selected_updates=updates,
                scheduler_total_updates=s["num_epochs"] * per_epoch,
                scheduler_warmup_updates=int(s["warmup"] * per_epoch),
                input_tokens=updates * batch * s["seq_len"],
                supervised_targets=updates * batch * (s["seq_len"] - ignored),
                ignored_targets_per_sequence=ignored,
                supervised_targets_per_sequence=s["seq_len"] - ignored,
                frames_per_sequence=math.ceil(s["seq_len"] / grid_len),
                grid_tokens=grid_len, input_output_vocab=s["vocab_size"],
                used_symbol_count=s["num_colors"] ** (s["patch"] ** 2) + 2)


def text_settings(smoke: bool = False) -> dict:
    values = dict(seed=5, lr=5e-4, pretrain=1, warmup=0.1, epochs=1,
                  batch_size=16, gradient_accumulation_steps=32,
                  pt_seq_len=1024, pt_vocab_size=64000, n_layer=24, n_head=32,
                  n_embd=2048, mixed_precision="fp16", log_grad=1,
                  log_grad_freq=100, grad_clip=1.0, grad_clip_enable=1,
                  weight_decay=1e-4, weight_tying=0, val_freq=100)
    if smoke:
        values.update(batch_size=2, gradient_accumulation_steps=2,
                      pt_seq_len=76, pt_vocab_size=18, n_layer=2, n_head=2,
                      n_embd=32, mixed_precision="none", val_freq=1)
    return values


def argv_from_settings(settings: dict) -> list[str]:
    return [item for key, value in settings.items() for item in ("--" + key, str(value))]


def make_plan(stage: str, *, smoke=False, seed=None, device="cuda") -> dict:
    if stage not in {"source", "owt", "scratch"}:
        raise ValueError(f"Unsupported stage: {stage}")
    s = source_settings(smoke) if stage == "source" else text_settings(smoke)
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        s["seed"] = seed
    if not smoke and device != "cuda":
        raise ValueError("The released-scale replay requires CUDA; use --smoke for CPU tests")
    if stage == "scratch":
        s["pretrain"] = 0
    args = argv_from_settings(s)
    if stage == "source":
        args += ["--model_type", "llama", "--model_name", "llama-large", "--token",
                 "--generate_train", "--interval_save", "--intervals", "2" if smoke else "10"]
        if not smoke:
            args += ["--filter_rules", "--autocast", "--log_grad", "--distributed"]
    else:
        args += ["--model_type", "llama", "--freeze_modules", "", "--reinit_modules", "embed", "none",
                 "--reinit_layer_idxs", "0", str(s["n_layer"]), "--interval_save"]
        if not smoke:
            args += ["--autocast"]
    args += ["--device", device if stage == "source" else "0"]
    return dict(schema_version=1, profile="smoke" if smoke else "released-code",
                stage=stage, requested_device=device, upstream_commit=LOCK["commit"], settings=s, argv=args,
                source_budget=source_budget(source_settings(smoke), 2 if smoke else 10),
                fidelity="smoke_only" if smoke else "released_code_replay_with_disclosed_runtime_repairs",
                exact_paper_reproduction=False, discrepancies=DISCREPANCIES,
                operational_changes=["No external W&B upload; local JSONL telemetry instead.",
                    "No implicit resume: fresh output directory required.",
                    "Stop after the recommended source checkpoint without rescaling its scheduler."])


def validate_data_manifest(root: Path, *, smoke=False) -> dict:
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("corpus") != "openwebtext":
        raise ValueError("Expected an OpenWebText manifest, not the FineWeb pilot corpus")
    if manifest.get("dtype") != "<u2" or manifest.get("tokenizer") != "tiktoken:gpt2":
        raise ValueError("Upstream expects little-endian uint16 GPT-2 tokens")
    origins = {"declared_author_assets", "reconstructed"} | ({"smoke"} if smoke else set())
    if manifest.get("origin") not in origins or not manifest.get("provenance"):
        raise ValueError("Missing explicit data origin/provenance")
    for split, filename in (("train", "train.bin"), ("validation", "val.bin")):
        meta = manifest["splits"][split]
        if meta["file"] != filename or meta["tokens"] < 2:
            raise ValueError("Invalid upstream data filename/token count")
        path = root / filename
        if path.stat().st_size != meta["tokens"] * 2 or sha256_file(path) != meta["sha256"]:
            raise ValueError(f"Data integrity failure: {split}")
    if not smoke and manifest["splits"]["train"]["tokens"] < 8_000_000_000:
        raise ValueError("Refusing to call a small corpus the published ~9B-token OWT replay; use --smoke for engineering tests")
    return manifest
