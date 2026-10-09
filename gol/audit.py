"""Read-only, held-out GoL loss audit for legacy and new source checkpoints.

No text corpus, optimizer update, parameter probe or checkpoint rewrite is used.
All predictors are scored on the SAME generated examples and target positions.
"""
import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config, LifeConfig
from .data import sha256_file
from .life import ALIVE, BOS, DEAD, EOS, FRAME, VOCAB_SIZE, sample, step
from .model import Decoder

BASELINES = ("unigram", "position_only", "previous_same_cell", "same_cell_left_up")


@dataclass(frozen=True)
class Layout:
    """Roles of targets at absolute positions 1..S, including packed restarts."""
    positions: np.ndarray
    cell: np.ndarray
    initial: np.ndarray
    evolved: np.ndarray
    frame: np.ndarray
    row: np.ndarray
    col: np.ndarray
    delimiter: np.ndarray
    frame_start: np.ndarray
    stride: int


def layout(config: LifeConfig, length: int) -> Layout:
    if length <= 0:
        raise ValueError("Target length must be positive")
    cells = config.board_size ** 2
    stride = cells + 1
    period = 2 + config.frames * stride
    positions = np.arange(1, length + 1)
    local = positions % period
    frame = (local - 1) // stride
    index = (local - 1) % stride
    cell = (local >= 1) & (local <= config.frames * stride) & (index < cells)
    initial = cell & (frame == 0)
    evolved = cell & (frame > 0)
    delimiter = np.full(length, FRAME, dtype=np.int64)
    delimiter[local == 0] = BOS
    delimiter[local == period - 1] = EOS
    return Layout(positions, cell, initial, evolved, frame, index // config.board_size,
                  index % config.board_size, delimiter, positions - index, stride)


def examples(cfg: Config, start: int, count: int, seed: int, batch_size: int):
    for offset in range(start, start + count, batch_size):
        stop = min(offset + batch_size, start + count)
        yield np.stack([sample(i, seed, cfg.model.seq_len + 1, cfg.life) for i in range(offset, stop)])


def features(tokens: np.ndarray, roles: Layout, name: str) -> np.ndarray:
    """Every feature at target j reads only positions strictly before j.

    Separator identities follow the configured grammar, NOT the target array.
    Current-frame top/left never wrap around to unseen bottom/right cells.
    """
    shape = (len(tokens), len(roles.positions))
    if name == "unigram":
        return np.zeros(shape, dtype=np.int64)
    if name == "position_only":
        return np.broadcast_to(np.arange(shape[1]), shape)
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    codes = np.broadcast_to(np.where(roles.evolved, 64, 0), shape).copy()
    ix = np.flatnonzero(roles.evolved)
    codes[:, ix] += tokens[:, roles.positions[ix] - roles.stride]
    if name == "same_cell_left_up":
        ix = np.flatnonzero(roles.cell & (roles.col > 0))
        codes[:, ix] += 2 + 2 * tokens[:, roles.positions[ix] - 1]
        ix = np.flatnonzero(roles.cell & (roles.row > 0))
        side = int(round((roles.stride - 1) ** 0.5))
        codes[:, ix] += 8 + 8 * tokens[:, roles.positions[ix] - side]
    codes[:, ~roles.cell] = 128 + roles.delimiter[~roles.cell]
    return codes


def fit_baselines(cfg: Config, count: int, seed: int, batch_size: int, alpha: float):
    if count <= 0 or batch_size <= 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Fit count, batch size and finite smoothing must be positive")
    roles = layout(cfg.life, cfg.model.seq_len)
    sizes = {"unigram": 1, "position_only": cfg.model.seq_len,
             "previous_same_cell": 134, "same_cell_left_up": 134}
    counts = {name: np.full((size, VOCAB_SIZE), alpha, dtype=np.float64) for name, size in sizes.items()}
    for tokens in examples(cfg, 0, count, seed, batch_size):
        targets = tokens[:, 1:]
        for name in BASELINES:
            key = features(tokens, roles, name) * VOCAB_SIZE + targets
            counts[name] += np.bincount(key.ravel(), minlength=counts[name].size).reshape(counts[name].shape)
    return {name: value / value.sum(1, keepdims=True) for name, value in counts.items()}


def masks(targets: np.ndarray, roles: Layout) -> dict[str, np.ndarray]:
    groups = {"all": np.ones_like(targets, dtype=bool), "initial_cells": roles.initial,
              "evolved_cells": roles.evolved, "delimiters": ~roles.cell,
              "evolved_alive": roles.evolved & (targets == ALIVE),
              "evolved_dead": roles.evolved & (targets == DEAD)}
    for frame in sorted(set(roles.frame[roles.cell].tolist())):
        groups[f"frame_{frame}_cells"] = roles.cell & (roles.frame == frame)
    return {name: np.broadcast_to(mask, targets.shape) for name, mask in groups.items()}


class Totals:
    def __init__(self):
        self.values = {}

    def add(self, losses: np.ndarray, targets: np.ndarray, roles: Layout):
        if losses.shape != targets.shape or not np.isfinite(losses).all():
            raise ValueError("Expected finite per-target losses with matching shape")
        for name, mask in masks(targets, roles).items():
            entry = self.values.setdefault(name, [0.0, 0])
            entry[0] += float(losses[mask].sum(dtype=np.float64))
            entry[1] += int(mask.sum())

    def result(self):
        return {name: {"nll_sum": total, "tokens": count, "loss": total / count if count else None}
                for name, (total, count) in self.values.items()}


def rule_reference_losses(tokens: np.ndarray, cfg: Config, roles: Layout) -> np.ndarray:
    """Known-rule reference, not a learned predictor or Bayes-optimal floor.

    Initial cells use the configured mean density, without hidden-density access.
    Evolved cells are predicted from a complete, causally visible previous frame.
    Separators follow the grammar. Epsilon clipping keeps failure cases finite.
    """
    prediction = np.broadcast_to(roles.delimiter, tokens[:, 1:].shape).copy()
    for start in np.unique(roles.frame_start[roles.evolved]):
        selected = np.flatnonzero(roles.evolved & (roles.frame_start == start))
        previous = tokens[:, start - roles.stride:start - 1].reshape(-1, cfg.life.board_size, cfg.life.board_size)
        evolved = np.stack([step(board, cfg.life.boundary).ravel() for board in previous])
        prediction[:, selected] = evolved[:, roles.positions[selected] - start]
    correct = prediction == tokens[:, 1:]
    probability = np.where(correct, 1.0, 1e-12)
    density = (cfg.life.density_min + cfg.life.density_max) / 2
    probability[:, roles.initial] = np.where(tokens[:, 1:][:, roles.initial] == ALIVE, density, 1 - density)
    return -np.log(probability)


def load_source(path: Path):
    """Accept old schema-1 warmup.pt files, but reject post-transfer checkpoints.

    Read-only mmap avoids materializing the unused Adam tensors. Never fall back
    to unrestricted unpickling; only load trusted local run artifacts.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta, state = payload["metadata"], payload["state"]
    if meta.get("schema_version") != 1 or meta.get("arm") != "gol_reset":
        raise ValueError(f"{path}: expected a schema-1 gol_reset checkpoint")
    if state.get("phase_index") != 0 or meta["stages"][0].get("source") != "gol":
        raise ValueError(f"{path}: expected a source-stage checkpoint, not a post-transfer text model")
    cfg = Config.from_dict(meta["config"])
    if tuple(payload["model"]["embedding.weight"].shape) != (VOCAB_SIZE, cfg.model.dim):
        raise ValueError(f"{path}: source embedding must have exactly five symbols")
    return payload, cfg


@torch.inference_mode()
def score_model(model: Decoder, cfg: Config, start: int, count: int, seed: int,
                batch_size: int, device: torch.device, dtype: str) -> dict:
    roles, total = layout(cfg.life, cfg.model.seq_len), Totals()
    model = model.to(device).eval()
    for tokens in examples(cfg, start, count, seed, batch_size):
        values = torch.from_numpy(tokens).to(device)
        context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()
        with context:
            logits = model(values[:, :-1])
        nll = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), values[:, 1:].reshape(-1), reduction="none")
        total.add(nll.reshape(len(tokens), -1).cpu().numpy(), tokens[:, 1:], roles)
    return total.result()


def audit(checkpoints, out, config=None, fit_sequences=4096, eval_sequences=4096,
          batch_size=8, seed=99173, device="auto", dtype="auto", alpha=0.5):
    if min(fit_sequences, eval_sequences, batch_size) <= 0 or seed < 0:
        raise ValueError("Positive sample counts/batch size and a nonnegative seed are required")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Smoothing alpha must be finite and positive")
    paths = [Path(p).resolve(strict=True) for p in checkpoints]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate checkpoint path")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; use a new audit directory")
    if device not in {"auto", "cpu", "cuda"} or dtype not in {"auto", "float32", "bfloat16"}:
        raise ValueError("Unsupported device/dtype")
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    resolved_dtype = ("bfloat16" if resolved_device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32") if dtype == "auto" else dtype
    if resolved_dtype == "bfloat16" and (resolved_device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("Use float32 on CPU or on GPUs without BF16 support")
    if resolved_device.type == "cpu":
        torch.set_num_threads(1)
    infos = []
    for path in paths:
        payload, candidate = load_source(path)
        if config is None:
            config = candidate
        if (candidate.life != config.life or candidate.model.seq_len != config.model.seq_len):
            raise ValueError("All checkpoints must share the same GoL generator and sequence length")
        if seed == candidate.train.data_seed + 1:
            raise ValueError("Audit seed collides with a source-training data seed")
        infos.append({"path": str(path), "sha256": sha256_file(path), "seed": payload["metadata"]["seed"],
                      "source_updates": payload["state"]["step"],
                      "training_git_revision": payload["metadata"].get("git_revision"),
                      "training_code_sha256": payload["metadata"].get("code_sha256"),
                      "training_config": payload["metadata"]["config"]})
        del payload
    if config is None:
        raise ValueError("Supply checkpoints or --config for a baselines-only audit")
    if seed == config.train.data_seed + 1:
        raise ValueError("Audit seed collides with the configured source-training data seed")
    roles = layout(config.life, config.model.seq_len)
    probabilities = fit_baselines(config, fit_sequences, seed, batch_size, alpha)
    totals = {name: Totals() for name in (*BASELINES, "known_rule_reference")}
    for tokens in examples(config, fit_sequences, eval_sequences, seed, batch_size):
        targets = tokens[:, 1:]
        for name in BASELINES:
            losses = -np.log(probabilities[name][features(tokens, roles, name), targets])
            totals[name].add(losses, targets, roles)
        totals["known_rule_reference"].add(rule_reference_losses(tokens, config, roles), targets, roles)
    records = [{"kind": "reference" if name == "known_rule_reference" else "baseline", "name": name,
                "groups": total.result()} for name, total in totals.items()]
    for path, info in zip(paths, infos):
        print(f"Evaluating {path} on {eval_sequences} held-out source sequences", flush=True)
        payload, candidate = load_source(path)
        model = Decoder(candidate.model, VOCAB_SIZE)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        groups = score_model(model, candidate, fit_sequences, eval_sequences, seed, batch_size, resolved_device, resolved_dtype)
        records.append({"kind": "checkpoint", "name": f"{path.parent.name}/{path.name}", **info, "groups": groups})
        del model
    report = {"schema_version": 1, "units": "nats per prediction target", "seed": seed,
              "fit_index_range": [0, fit_sequences], "eval_index_range": [fit_sequences, fit_sequences + eval_sequences],
              "ranges_are": "half-open, same seed, disjoint fit/eval indices",
              "life": asdict(config.life), "seq_len": config.model.seq_len, "smoothing_alpha": alpha,
              "targets_per_sequence": {"initial_cells": int(roles.initial.sum()), "evolved_cells": int(roles.evolved.sum()),
                                       "delimiters": int((~roles.cell).sum())},
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                          "device": str(resolved_device), "dtype": resolved_dtype, "batch_size": batch_size},
              "audit_source_sha256": sha256_file(Path(__file__)),
              "generator_source_sha256": sha256_file(Path(__file__).with_name("life.py")),
              "model_source_sha256": sha256_file(Path(__file__).with_name("model.py")), "records": records,
              "limitations": ["Teacher-forced loss does not by itself establish rule learning or language transfer.",
                              "Known-rule reference uses a fixed mean-density initial predictor; it is not Bayes optimal.",
                              "Old warmup training loss is not a matched held-out baseline.",
                              "This operation does not train, resume, or modify any checkpoint."]}
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (out / "losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "name", "checkpoint", "seed", "source_updates", "group", "tokens", "nll_sum", "loss"])
        for record in records:
            for group, value in record["groups"].items():
                writer.writerow([record["kind"], record["name"], record.get("path", ""), record.get("seed", ""),
                                 record.get("source_updates", ""), group, value["tokens"], value["nll_sum"], value["loss"]])
    for record in records:
        print(f'{record["name"]}: all={record["groups"]["all"]["loss"]:.6f}, '
              f'evolved={record["groups"]["evolved_cells"]["loss"]:.6f}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", type=Path, default=[])
    parser.add_argument("--config", type=Path, help="Only needed for baselines without checkpoints; otherwise metadata is used")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fit-sequences", type=int, default=4096)
    parser.add_argument("--eval-sequences", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    audit(args.checkpoints, args.out, Config.load(args.config) if args.config else None,
          args.fit_sequences, args.eval_sequences, args.batch_size, args.seed, args.device, args.dtype, args.alpha)


if __name__ == "__main__":
    main()
