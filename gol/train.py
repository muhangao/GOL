"""Two-stage loss experiments with paired initialization, DDP and resumable updates."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import Config, flops_per_token, learning_rate, warmup_steps, phase_train_config
from .data import TextCorpus
from .life import VOCAB_SIZE, sample as life_sample
from .model import build_model, transfer_body, fingerprint

ARMS = ("scratch", "text_reset", "gol_reset", "text_continue")


class Runtime:
    def __init__(self, device="auto"):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(f"cuda:{local_rank}" if device == "cuda" else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        else:
            torch.set_num_threads(1)
        self.owns_group = self.world > 1 and not dist.is_initialized()
        if self.owns_group:
            dist.init_process_group("nccl" if device == "cuda" else "gloo")

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reduce(self, tensor):
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor

    def wrap(self, model):
        model = model.to(self.device)
        if self.world == 1:
            return model
        return DDP(model, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                   broadcast_buffers=False)

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def optimizer_for(model, cfg, device):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")


def autocast(cfg, device):
    if cfg.dtype == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def plan(config, arm, text_vocab):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    stages = []
    if arm != "scratch":
        vocab = VOCAB_SIZE if arm == "gol_reset" else text_vocab
        source_cfg = phase_train_config(config, arm, "warmup")
        stages.append({"name": "warmup", "source": "gol" if arm == "gol_reset" else "warmup",
                       "vocab": vocab, "steps": warmup_steps(config, text_vocab, vocab, source_cfg.global_batch_size)})
    stages.append({"name": "text", "source": "train", "vocab": text_vocab,
                   "steps": config.train.text_steps})
    for stage in stages:
        stage["training"] = asdict(phase_train_config(config, arm, stage["name"]))
    return stages


def batch(corpus, source, indices, cfg, seed, device):
    length = cfg.model.seq_len + 1
    samples = [life_sample(i, seed, length, cfg.life) if source == "gol"
               else corpus.sample(source, i, seed, length) for i in indices]
    values = torch.from_numpy(np.stack(samples)).to(device)
    return values[:, :-1].contiguous(), values[:, 1:].contiguous()


@torch.no_grad()
def evaluate(model, corpus, cfg, runtime):
    # Use the unwrapped module: ranks may have unequal validation batch counts.
    model = unwrap(model)
    model.eval()
    indices = list(range(runtime.rank, cfg.train.eval_sequences, runtime.world))
    stats = torch.zeros(2, device=runtime.device, dtype=torch.float64)
    micro = cfg.train.micro_batch_size
    for offset in range(0, len(indices), micro):
        selected = indices[offset:offset + micro]
        x, y = batch(corpus, "valid", selected, cfg, cfg.train.data_seed + 3, runtime.device)
        with autocast(cfg.train, runtime.device):
            loss = model(x, y)
        stats[0] += loss.double() * y.numel()
        stats[1] += y.numel()
    runtime.reduce(stats)
    result = (stats[0] / stats[1]).item()
    if not np.isfinite(result):
        raise FloatingPointError("Non-finite text validation loss")
    return result


def update(model, optimizer, corpus, stage, step, cfg, runtime, lr, train_cfg=None):
    train_cfg = train_cfg or cfg.train
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = train_cfg.micro_batch_size
    accumulation = train_cfg.global_batch_size // (micro * runtime.world)
    data_seed = train_cfg.data_seed + (1 if stage["name"] == "warmup" else 2)
    mean_loss = torch.zeros((), device=runtime.device)
    for group in optimizer.param_groups:
        group["lr"] = lr
    for k in range(accumulation):
        first = ((step - 1) * train_cfg.global_batch_size
                 + (k * runtime.world + runtime.rank) * micro)
        x, y = batch(corpus, stage["source"], range(first, first + micro), cfg, data_seed, runtime.device)
        sync = model.no_sync() if isinstance(model, DDP) and k + 1 < accumulation else nullcontext()
        with sync:
            with autocast(train_cfg, runtime.device):
                loss = model(x, y)
            (loss / accumulation).backward()
        mean_loss += loss.detach() / accumulation
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return (runtime.reduce(mean_loss) / runtime.world).item()


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_experiment(config, data, out, arm, seed=0, device="auto", resume=False,
                   stop_after=None, verify_data=False):
    """Run or resume one arm. `stop_after` is an absolute optimizer-update count.

    Checkpoints are written only at complete updates. Training contains no
    dropout and all data sampling is stateless, so there is no stochastic
    training RNG to restore. Exact CPU resumption is tested; CUDA kernels may
    remain nondeterministic. Load only checkpoints produced by trusted runs.
    """
    runtime = Runtime(device)
    try:
        return _run(config, data, Path(out), arm, seed, runtime, resume, stop_after, verify_data)
    finally:
        runtime.close()


def _run(cfg, data, out, arm, seed, rt, resume, stop_after, verify_data):
    if seed < 0 or (stop_after is not None and stop_after < 0):
        raise ValueError("seed and stop_after must be nonnegative")
    for phase in (("text",) if arm == "scratch" else ("warmup", "text")):
        phase_cfg = phase_train_config(cfg, arm, phase)
        if phase_cfg.global_batch_size % (phase_cfg.micro_batch_size * rt.world):
            raise ValueError(f"{phase}: global_batch_size must divide evenly by micro_batch_size * WORLD_SIZE")
    if cfg.train.dtype == "bfloat16" and (rt.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("This BF16 configuration requires a BF16-capable CUDA GPU; use float32 for CPU")
    corpus = TextCorpus(data, verify=verify_data)
    for split, values in corpus.arrays.items():
        if len(values) <= cfg.model.seq_len:
            raise ValueError(f"{split} must contain at least seq_len + 1 tokens")
    stages = plan(cfg, arm, corpus.vocab_size)
    metadata = {"schema_version": 1, "arm": arm, "seed": seed, "config": cfg.to_dict(),
                "manifest_sha256": corpus.fingerprint, "tokenizer": corpus.manifest["tokenizer"],
                "world_size": rt.world, "device_type": rt.device.type, "torch": str(torch.__version__),
                "numpy": np.__version__, "python": platform.python_version(), "stages": stages,
                "code_sha256": code_fingerprint(), "git_revision": git_revision(),
                "flops_estimator": "dense_matmul_fwd_bwd_v1; excludes non-matmul and auxiliary work"}
    state = {"phase_index": 0, "step": 0, "started": False, "updates": 0,
             "warmup_tokens": 0, "text_tokens": 0, "training_flops_est": 0,
             "update_seconds": 0.0, "metric_count": 0, "last_text_val_loss": None}
    checkpoint = None
    if resume:
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["metadata"] != metadata:
            raise ValueError("Resume metadata mismatch: keep code, config, corpus, seed, arm and runtime unchanged")
        state = checkpoint["state"]
        if state["phase_index"] == len(stages):
            if rt.rank == 0:
                atomic_json(out / "complete.json", state)
            return state
        # Remove a possibly partial/uncommitted log tail after an interrupted save.
        if rt.rank == 0:
            lines = (out / "metrics.jsonl").read_text().splitlines()
            if len(lines) < state["metric_count"]:
                raise ValueError("Metrics log is shorter than the checkpoint")
            committed = lines[:state["metric_count"]]
            for line in committed:
                json.loads(line)
            (out / "metrics.jsonl").write_text("\n".join(committed) + ("\n" if committed else ""))
    else:
        # Check before any rank writes, then broadcast the decision to avoid a
        # race where another rank mistakes newly written metadata for old data.
        occupied = torch.tensor(int(out.exists() and any(out.iterdir())) if rt.rank == 0 else 0,
                                device=rt.device)
        if rt.world > 1:
            dist.broadcast(occupied, src=0)
        if occupied.item():
            raise FileExistsError(f"Refusing to overwrite {out}; use --resume or a new run directory")
        if rt.rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "metadata.json", metadata)
    rt.barrier()
    stage = stages[state["phase_index"]]
    io_seed = seed + 2 if stage["name"] == "text" and arm != "text_continue" else seed + 1
    base = build_model(cfg.model, stage["vocab"], seed, io_seed)
    if checkpoint is not None:
        base.load_state_dict(checkpoint["model"])
    elif rt.rank == 0:
        atomic_json(out / "initialization.json", {"body_sha256": fingerprint(base),
                    "io_sha256": fingerprint(base, io=True),
                    "parameters": sum(p.numel() for p in base.parameters())})
    model = rt.wrap(base)
    optimizer = optimizer_for(model, phase_train_config(cfg, arm, stage["name"]), rt.device)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    del checkpoint, base

    def save(name="last.pt"):
        if rt.rank == 0:
            payload = {"metadata": metadata, "state": dict(state),
                       "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict()}
            tmp = out / (name + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(out / name)
        rt.barrier()

    def log(train_loss=None, lr=None):
        stage_now = stages[state["phase_index"]]
        # GoL uses a different vocabulary: text loss is undefined before transfer.
        val = None if stage_now["source"] == "gol" else evaluate(model, corpus, cfg, rt)
        if stage_now["name"] == "text":
            state["last_text_val_loss"] = val
        active_cfg = phase_train_config(cfg, arm, stage_now["name"])
        row = {"phase": stage_now["name"],
               "global_batch_size": active_cfg.global_batch_size,
               "micro_batch_size": active_cfg.micro_batch_size, "phase_step": state["step"],
               "updates": state["updates"], "train_loss": train_loss, "text_val_loss": val,
               "learning_rate": lr, "warmup_tokens": state["warmup_tokens"],
               "text_tokens": state["text_tokens"],
               "total_tokens": state["warmup_tokens"] + state["text_tokens"],
               "training_flops_est": state["training_flops_est"], "update_seconds": state["update_seconds"]}
        if rt.rank == 0:
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({"arm": arm, "seed": seed, **row}, allow_nan=False), flush=True)
        state["metric_count"] += 1

    while state["phase_index"] < len(stages):
        stage = stages[state["phase_index"]]
        active_cfg = phase_train_config(cfg, arm, stage["name"])
        tokens_per_update = active_cfg.global_batch_size * cfg.model.seq_len
        if not state["started"]:
            log()
            state["started"] = True
            save()
        if stop_after is not None and state["updates"] >= stop_after:
            save()
            return state
        while state["step"] < stage["steps"]:
            step = state["step"] + 1
            if arm == "text_continue":
                lr_step, horizon = state["updates"] + 1, sum(s["steps"] for s in stages)
            else:
                lr_step, horizon = step, stage["steps"]
            lr = learning_rate(lr_step, horizon, active_cfg)
            rt.synchronize()
            started = time.perf_counter()
            loss = update(model, optimizer, corpus, stage, step, cfg, rt, lr, active_cfg)
            rt.synchronize()
            state["update_seconds"] += time.perf_counter() - started
            state["step"] = step
            state["updates"] += 1
            key = "warmup_tokens" if stage["name"] == "warmup" else "text_tokens"
            state[key] += tokens_per_update
            state["training_flops_est"] += tokens_per_update * flops_per_token(cfg.model, stage["vocab"])
            pause = stop_after is not None and state["updates"] >= stop_after
            finished = step == stage["steps"]
            if step == 1 or step % cfg.train.eval_every == 0 or finished or pause:
                log(loss, lr)
            if step % cfg.train.checkpoint_every == 0 or finished or pause:
                save()
            if pause:
                if finished and stage["name"] == "warmup":
                    save("warmup.pt")
                return state
        if stage["name"] == "warmup":
            save("warmup.pt")
            before = unwrap(model)
            report = {"body_before": fingerprint(before), "io_before": fingerprint(before, io=True),
                      "optimizer_reset": arm != "text_continue", "io_reset": arm != "text_continue"}
            if arm != "text_continue":
                after = transfer_body(before, corpus.vocab_size, seed, seed + 2)
                model = rt.wrap(after)
                optimizer = optimizer_for(model, cfg.train, rt.device)
                if len(optimizer.state):
                    raise RuntimeError("New optimizer unexpectedly has accumulated state")
                del after
            report.update({"body_after": fingerprint(unwrap(model)), "io_after": fingerprint(unwrap(model), io=True)})
            if report["body_before"] != report["body_after"]:
                raise RuntimeError("Transfer unexpectedly changed Transformer-body weights")
            if rt.rank == 0:
                atomic_json(out / "transfer.json", report)
            del before
        state["phase_index"] += 1
        state["step"] = 0
        state["started"] = False
    save()
    if rt.rank == 0:
        atomic_json(out / "complete.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true", help="Print resolved stages/budgets without training or creating a run")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Pause after this absolute number of complete optimizer updates")
    parser.add_argument("--verify-data", action="store_true", help="Recompute the full prepared-token checksums")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.plan_only:
        corpus = TextCorpus(args.data)
        stages = plan(config, args.arm, corpus.vocab_size)
        for stage in stages:
            tokens = stage["steps"] * stage["training"]["global_batch_size"] * config.model.seq_len
            stage["token_exposures"] = tokens
            stage["training_flops_est"] = tokens * flops_per_token(config.model, stage["vocab"])
        print(json.dumps(stages, indent=2))
        return
    if args.out is None:
        parser.error("--out is required unless --plan-only is used")
    run_experiment(config, args.data, args.out, args.arm, args.seed,
                   args.device, args.resume, args.stop_after, args.verify_data)


if __name__ == "__main__":
    main()
