from dataclasses import replace
import json
import subprocess
import sys
import pytest
import torch

from gol.data import TextCorpus
from gol.train import ARMS, plan, run_experiment
from gol.plot import plot


def load(path):
    return torch.load(path / "last.pt", map_location="cpu", weights_only=True)


@pytest.mark.parametrize("arm", ARMS)
def test_end_to_end_and_optimizer_reset(config, corpus, tmp_path, arm):
    out = tmp_path / arm
    state = run_experiment(config, corpus, out, arm, device="cpu")
    stages = plan(config, arm, TextCorpus(corpus).vocab_size)
    assert state["updates"] == sum(s["steps"] for s in stages)
    assert state["text_tokens"] == config.train.text_steps * config.train.global_batch_size * config.model.seq_len
    assert (out / "complete.json").exists()
    checkpoint = load(out)
    expected = state["updates"] if arm == "text_continue" else config.train.text_steps
    assert all(int(s["step"]) == expected for s in checkpoint["optimizer"]["state"].values())
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    text_rows = [r for r in rows if r["phase"] == "text"]
    assert text_rows[0]["phase_step"] == 0 and text_rows[0]["text_tokens"] == 0
    assert all(torch.isfinite(torch.tensor(r["text_val_loss"])) for r in text_rows)
    if arm != "scratch":
        report = json.loads((out / "transfer.json").read_text())
        assert report["body_before"] == report["body_after"]
        assert report["optimizer_reset"] == (arm != "text_continue")
    if arm == "gol_reset":
        assert all(r["text_val_loss"] is None for r in rows if r["phase"] == "warmup")
    with pytest.raises(FileExistsError):
        run_experiment(config, corpus, out, arm, device="cpu")
    # Completed runs are safe to resume and do not append duplicate metrics.
    before = (out / "metrics.jsonl").read_bytes()
    run_experiment(config, corpus, out, arm, device="cpu", resume=True)
    assert before == (out / "metrics.jsonl").read_bytes()


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("pause_location", ("early", "boundary", "text"))
def test_exact_cpu_resume(config, corpus, tmp_path, arm, pause_location):
    stages = plan(config, arm, TextCorpus(corpus).vocab_size)
    warmup = stages[0]["steps"] if len(stages) > 1 else 0
    stop = {"early": 1, "boundary": max(1, warmup), "text": warmup + 1}[pause_location]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_experiment(config, corpus, full, arm, seed=7, device="cpu")
    run_experiment(config, corpus, resumed, arm, seed=7, device="cpu", stop_after=stop)
    assert not (resumed / "complete.json").exists()
    with (resumed / "metrics.jsonl").open("a") as handle:
        handle.write('{"partial_uncommitted_tail":')
    run_experiment(config, corpus, resumed, arm, seed=7, device="cpu", resume=True)
    a, b = load(full), load(resumed)
    assert a["state"]["last_text_val_loss"] == b["state"]["last_text_val_loss"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    assert a["state"]["training_flops_est"] == b["state"]["training_flops_est"]


def test_paired_io_and_loss_plots(config, corpus, tmp_path):
    for arm in ARMS:
        run_experiment(config, corpus, tmp_path / "runs" / arm, arm, device="cpu")
    scratch = json.loads((tmp_path / "runs/scratch/initialization.json").read_text())
    for arm in ("text_reset", "gol_reset"):
        init = json.loads((tmp_path / f"runs/{arm}/initialization.json").read_text())
        report = json.loads((tmp_path / f"runs/{arm}/transfer.json").read_text())
        assert init["body_sha256"] == scratch["body_sha256"]
        assert report["io_after"] == scratch["io_sha256"]
    pytest.importorskip("matplotlib")
    plot(tmp_path / "runs", tmp_path / "plots")
    assert (tmp_path / "plots/loss_vs_total_flops.png").stat().st_size > 1000
    assert (tmp_path / "plots/loss_vs_text_tokens.png").stat().st_size > 1000
    assert len((tmp_path / "plots/final_losses.csv").read_text().splitlines()) == 5


def test_resume_mismatch(config, corpus, tmp_path):
    out = tmp_path / "run"
    run_experiment(config, corpus, out, "gol_reset", device="cpu", stop_after=1)
    other = replace(config, train=replace(config.train, lr=0.01))
    with pytest.raises(ValueError, match="metadata mismatch"):
        run_experiment(other, corpus, out, "gol_reset", device="cpu", resume=True)


def test_two_rank_cpu_ddp(config, corpus, tmp_path):
    # Five eval sequences deliberately do not divide by two ranks.
    cfg = replace(config, train=replace(config.train, text_steps=2))
    config_path = tmp_path / "ddp_config.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    single = tmp_path / "single"
    multi = tmp_path / "multi"
    run_experiment(cfg, corpus, single, "gol_reset", device="cpu")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
               "-m", "gol.train", "--config", str(config_path), "--data", str(corpus),
               "--out", str(multi), "--arm", "gol_reset", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    a, b = load(single), load(multi)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=2e-4, atol=2e-5)
    assert abs(a["state"]["last_text_val_loss"] - b["state"]["last_text_val_loss"]) < 1e-5
