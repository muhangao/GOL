"""Execute the pinned authors' NCA/OWT drivers, not the GoL trainer.

Default invocation prints a plan. --execute opts into a fresh run. Production
runs also require acknowledgement of the documented paper/code discrepancies.
Original generator/model/optimizer/scheduler/loop code is imported from a
verified snapshot. Only the listed mechanical OWT repairs are applied.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time

from .patches import apply_repairs
from .protocol import (LOCK, make_plan, verify_checkout, dependency_report,
                       validate_data_manifest, sha256_file, write_json)


class SelectedCheckpointReached(Exception):
    """Normal termination after the recommended checkpoint has been saved."""


def snapshot(root: Path, output: Path, stage: str) -> tuple[Path, list[dict]]:
    target = output / "upstream_snapshot"
    target.mkdir()
    for rel in LOCK["files"]:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dest)
    return target, apply_repairs(target, stage)


def load_driver(snapshot_root: Path, stage: str):
    if "utils" in sys.modules:
        raise RuntimeError("Run replay in a fresh Python process: an unrelated 'utils' module is already loaded")
    sys.path.insert(0, str(snapshot_root))
    path = snapshot_root / "src" / ("nca_ppt.py" if stage == "source" else "openwebtext_pt.py")
    spec = importlib.util.spec_from_file_location("nca_released_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the verified upstream driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def body_hash(model) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith(("input_proj.", "output_proj.")):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def check_source_receipt(run: Path, *, smoke: bool) -> tuple[Path, dict]:
    run = Path(run).resolve(strict=True)
    receipt = json.loads((run / "selected_source.json").read_text())
    expected_profile = "smoke" if smoke else "released-code"
    if (receipt.get("upstream_commit") != LOCK["commit"] or receipt.get("profile") != expected_profile
            or receipt.get("kind") != "nca_source_checkpoint"):
        raise ValueError("The NCA handoff needs a matching, pinned source receipt, not a GoL checkpoint")
    relative = Path(receipt["checkpoint"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe checkpoint path in source receipt")
    checkpoint = (run / relative).resolve(strict=True)
    if not checkpoint.is_relative_to(run) or checkpoint.is_symlink():
        raise ValueError("Checkpoint must be contained in its source run")
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Source checkpoint content changed after selection")
    if receipt.get("input_tokens") != make_plan("source", smoke=smoke, device="cpu" if smoke else "cuda")["source_budget"]["input_tokens"]:
        raise ValueError("Selected source checkpoint has the wrong token budget")
    return checkpoint, receipt


class Telemetry:
    """Read-only instrumentation; no changes to logits, loss, gradients or RNG."""
    def __init__(self, output: Path, plan: dict):
        self.output, self.plan = output, plan
        self.model = self.optimizer = self.scheduler = None
        self.input_tokens = self.supervised_targets = self.optimizer_steps = 0
        self.expected_body = None
        self.last_native_iteration = None
        self.last_validation_loss = None
        self.lock = threading.Lock()

    def emit(self, event: str, **fields):
        row = dict(event=event, input_tokens=self.input_tokens,
                   supervised_targets=self.supervised_targets,
                   successful_optimizer_steps=self.optimizer_steps, **fields)
        with self.lock, (self.output / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False, default=str) + "\n")

    def attach(self, driver):
        import torch
        original_log = driver.wandb_log

        def log(data, args):
            clean = {str(k): (v.item() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
            if "iteration" in clean:
                self.last_native_iteration = int(clean["iteration"])
            self.emit("native_log", values=clean)
            return original_log(data, args)
        driver.wandb_log = log

        original_criterion = driver.CrossEntropyLoss
        observer = self

        class ObservedLoss(original_criterion):
            def forward(self, logits, targets):
                loss = super().forward(logits, targets)
                if observer.model is not None and observer.model.training:
                    observer.input_tokens += targets.numel()
                    observer.supervised_targets += int((targets != self.ignore_index).sum().item())
                return loss
        driver.CrossEntropyLoss = ObservedLoss

        original_scheduler = driver.get_lr_scheduler

        def make_scheduler(optimizer, *args, **kwargs):
            self.optimizer = optimizer
            self.scheduler = original_scheduler(optimizer, *args, **kwargs)
            optimizer.register_step_post_hook(lambda *_: self._step())
            assert self.model is not None
            actual_body = body_hash(self.model)
            if self.expected_body is not None and actual_body != self.expected_body:
                raise RuntimeError("Upstream handoff unexpectedly changed non-I/O weights")
            io = self.model.input_proj.weight
            head = self.model.output_proj.weight
            description = dict(parameters=sum(p.numel() for p in self.model.parameters()),
                               input_shape=list(io.shape), output_shape=list(head.shape),
                               tied=io.data_ptr() == head.data_ptr(), body_sha256=actual_body,
                               trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                               optimizer=type(optimizer).__name__, optimizer_defaults=optimizer.defaults,
                               scheduler_kwargs=kwargs, scheduler_args=list(args))
            if not self.plan["profile"] == "smoke":
                if io.shape[1] != 2048 or len(self.model.layers) != 24 or description["tied"]:
                    raise RuntimeError("Model is not the released-code architecture")
            write_json(self.output / "model_and_optimizer.json", description)
            return self.scheduler
        driver.get_lr_scheduler = make_scheduler

        if self.plan["stage"] == "source":
            original_builder = driver.build_model
            def build(args):
                self.model = original_builder(args)
                return self.model
            driver.build_model = build
            original_loader = driver.build_dataloader
            def loader(*a, **kw):
                result = original_loader(*a, **kw)
                import numpy as np
                seeds = np.asarray(kw["rule_seeds"])
                prefix = result.dataset.seq[:4].detach().cpu().contiguous().numpy()
                self.emit("source_loader", sequences=len(result.dataset),
                          rule_count=len(seeds), rule_seed_dtype=str(seeds.dtype),
                          rule_bank_sha256=hashlib.sha256(seeds.tobytes()).hexdigest(),
                          first_four_sequences_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
                          token_dtype=str(prefix.dtype), sequence_shape=list(result.dataset.seq.shape))
                return result
            driver.build_dataloader = loader
            original_rules = driver.generate_rules_batch
            banks = [0]
            def rules(*a, **kw):
                result = original_rules(*a, **kw)
                import numpy as np
                directory = self.output / "rule_banks"
                directory.mkdir(exist_ok=True)
                name = directory / f"bank_{banks[0]:03d}.npy"
                np.save(name, np.asarray(result), allow_pickle=False)
                banks[0] += 1
                self.emit("rule_bank", file=str(name.relative_to(self.output)),
                          sha256=sha256_file(name), count=len(result),
                          lower=kw.get("threshold"), upper=kw.get("upper_bound"), mode=kw.get("mode"))
                return result
            driver.generate_rules_batch = rules
        else:
            original_constructor = driver.DownstreamLlamaLM
            def construct(*args, **kwargs):
                self.model = original_constructor(*args, **kwargs)
                return self.model
            driver.DownstreamLlamaLM = construct
            original_load = driver.load_model
            def strict_load(model, *args, **kwargs):
                # Assertion only: a matching valid checkpoint loads identically.
                kwargs["strict"] = True
                model = original_load(model, *args, **kwargs)
                self.expected_body = body_hash(model)
                return model
            driver.load_model = strict_load

        original_val = driver.val_epoch
        def val(*args, **kwargs):
            value = original_val(*args, **kwargs)
            loss = value[0] / value[1] if isinstance(value, tuple) else value
            if not math.isfinite(float(loss)):
                raise FloatingPointError("Non-finite native validation loss")
            self.last_validation_loss = float(loss)
            self.emit("validation", loss=float(loss), aggregation="native_mean_of_batch_means")
            return value
        driver.val_epoch = val

    def _step(self):
        self.optimizer_steps += 1


def execute(args, plan: dict):
    import torch
    deps = dependency_report()
    mismatches = {k: v for k, v in deps.items() if not v["matches"]}
    if mismatches and not args.allow_environment_drift:
        raise RuntimeError("Dependency drift; use a separate pinned environment. " + json.dumps(mismatches))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no training was started")
    if not args.smoke and args.stage != "source" and torch.cuda.device_count() < 2:
        raise RuntimeError("This reference selects the original multi-GPU DataParallel path; allocate at least two GPUs. The released single-GPU compile path is not validated.")
    if not args.smoke and not args.accept_release_differences:
        raise ValueError("Read docs/NCA_REPRODUCTION.md, then acknowledge --accept-release-differences. This is code replay, not verified exact paper equivalence.")
    root = args.upstream.resolve(strict=True)
    verified = verify_checkout(root)
    output = args.out.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Keep run outputs and pristine upstream checkout in separate trees")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite or implicitly resume {output}")
    data = checkpoint = source_receipt = None
    if args.stage != "source":
        if args.data is None:
            raise ValueError("--data with a verified OWT manifest is required")
        data = validate_data_manifest(args.data, smoke=args.smoke)
        if args.stage == "owt":
            if args.source_run is None:
                raise ValueError("--source-run is required for NCA-to-OWT transfer")
            checkpoint, source_receipt = check_source_receipt(args.source_run, smoke=args.smoke)
    output.mkdir(parents=True)
    snap, repairs = snapshot(root, output, args.stage)
    runtime = dict(python=platform.python_version(), dependencies=deps,
                   environment_drift=bool(mismatches), platform=platform.platform(),
                   cuda=torch.version.cuda, cuda_device_count=torch.cuda.device_count(),
                   cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   default_cuda_autocast_dtype=str(torch.get_autocast_dtype("cuda")),
                   selected_environment={k: os.environ.get(k) for k in
                       ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_PYTHON_CLIENT_PREALLOCATE", "SLURM_JOB_ID")})
    write_json(output / "plan.json", plan)
    write_json(output / "provenance.json", dict(upstream_files=verified, repairs=repairs, runtime=runtime,
        wrapper_files={p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        data=data, source_receipt=source_receipt,
        exact_paper_reproduction=False,
        checkpoint_security="Only trusted checkpoints produced by this pinned replay may be loaded. Upstream uses pickle."))
    previous = Path.cwd()
    began = time.perf_counter()
    try:
        os.chdir(snap)
        os.environ["WANDB_MODE"] = "disabled"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        driver = load_driver(snap, args.stage)
        import jax
        if jax.config.x64_enabled:
            raise RuntimeError("JAX x64 changes gzip token bytes; disable it to replay the released default")
        write_json(output / "jax_runtime.json", dict(backend=jax.default_backend(), x64=False,
                   devices=[str(x) for x in jax.devices()], version=jax.__version__,
                   preallocate=os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]))
        native_argv = list(plan["argv"]) + ["--save_dir", str(output / "checkpoints")]
        if args.stage != "source":
            native_argv += ["--data_dir", str(args.data.resolve())]
            if checkpoint is not None:
                native_argv += ["--model_path", str(checkpoint.parent), "--model_file", checkpoint.name]
            parser = driver.create_openwebtext_parser()
            native_args = driver.args_to_dataclass(parser.parse_args(native_argv), is_v2l=False)
            if args.device == "cpu":
                native_args.device = "cpu"
        else:
            parser = driver.create_parser()
            native_args = driver.nca_args_to_dataclass(parser.parse_args(native_argv))
        write_json(output / "resolved_args.json", {k: v for k, v in vars(native_args).items()
                                                     if isinstance(v, (str, int, float, bool, list, type(None)))})
        telemetry = Telemetry(output, plan)
        telemetry.attach(driver)
        original_save = driver.save_checkpoint

        def save(*a, **kw):
            original_save(*a, **kw)
            # Original function signature: epoch,count,model,opt,sched,best,best_el,metrics,save_dir.
            epoch = a[0] if a else kw["epoch"]
            save_dir = Path(a[8] if len(a) > 8 else kw["save_dir"])
            chosen = plan["source_budget"]
            if (args.stage == "source" and epoch == chosen["selected_epoch"]
                    and save_dir.name == "interval_save" and not kw.get("best", False)):
                updates = kw.get("total_iterations", 0)
                if updates != chosen["selected_updates"] or telemetry.input_tokens != chosen["input_tokens"]:
                    raise RuntimeError("Observed source budget differs from the predeclared checkpoint budget")
                if telemetry.supervised_targets != chosen["supervised_targets"]:
                    raise RuntimeError("Observed supervision mask differs from the released source contract")
                selected = save_dir / f"model_{epoch}.pth"
                receipt = dict(kind="nca_source_checkpoint", upstream_commit=LOCK["commit"],
                    profile=plan["profile"], seed=native_args.seed, epoch=epoch,
                    native_iterations=updates, successful_optimizer_steps=telemetry.optimizer_steps,
                    input_tokens=telemetry.input_tokens, supervised_targets=telemetry.supervised_targets,
                    checkpoint=str(selected.relative_to(output)), checkpoint_sha256=sha256_file(selected),
                    plan_sha256=sha256_file(output / "plan.json"), source_validation_loss=telemetry.last_validation_loss)
                write_json(output / "selected_source.json", receipt)
                raise SelectedCheckpointReached()
        driver.save_checkpoint = save
        try:
            driver.main(native_args)
        except SelectedCheckpointReached:
            pass
        if args.stage == "source":
            if not (output / "selected_source.json").exists():
                raise RuntimeError("The original driver exited before saving the selected source checkpoint")
        else:
            # Native code does not necessarily save/evaluate the very last update.
            # Final-only observation does not change the preceding learning trajectory.
            val_loader, _ = driver.build_dataloader(native_args, "validation")
            driver.val_epoch(native_args, telemetry.model, val_loader, torch.nn.CrossEntropyLoss())
            torch.save({"model": telemetry.model.state_dict(), "optimizer": telemetry.optimizer.state_dict(),
                        "scheduler": telemetry.scheduler.state_dict(),
                        "successful_optimizer_steps": telemetry.optimizer_steps}, output / "final.pt")
        complete = dict(stage=args.stage, profile=plan["profile"], upstream_commit=LOCK["commit"],
                        exact_paper_reproduction=False, input_tokens=telemetry.input_tokens,
                        supervised_targets=telemetry.supervised_targets,
                        successful_optimizer_steps=telemetry.optimizer_steps,
                        final_validation_loss=telemetry.last_validation_loss,
                        seconds=time.perf_counter() - began)
        write_json(output / "complete.json", complete)
        return complete
    except Exception as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        os.chdir(previous)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("source", "owt", "scratch"), required=True)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--source-run", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--smoke", action="store_true", help="Explicit tiny engineering run, never a scientific result")
    p.add_argument("--execute", action="store_true", help="Without this flag, only print the plan")
    p.add_argument("--accept-release-differences", action="store_true")
    p.add_argument("--allow-environment-drift", action="store_true", help="Mark the run non-locked; never silently substitute versions")
    args = p.parse_args()
    plan = make_plan(args.stage, smoke=args.smoke, seed=args.seed, device=args.device)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if args.upstream is None or args.out is None:
        p.error("--execute requires --upstream and --out")
    print(json.dumps(execute(args, plan), indent=2))


if __name__ == "__main__":
    main()
