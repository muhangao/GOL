from dataclasses import replace
import json
import subprocess
import sys
import pytest
import torch

from gol.config import Config, WarmupConfig, flops_per_token, phase_train_config, warmup_steps
from gol.data import TextCorpus
from gol.train import ARMS, plan, run_experiment


def test_legacy_config_and_noop_override(config):
    assert "warmup" not in config.to_dict()
    assert Config.from_dict(config.to_dict()) == config
    changed = replace(config, warmup=WarmupConfig())
    for arm in ARMS:
        assert plan(config, arm, 257) == plan(changed, arm, 257)


@pytest.mark.parametrize("mode", ["tokens", "estimated_flops"])
@pytest.mark.parametrize("batch", [1, 2, 3, 8])
def test_batch_changes_updates_not_reference_budget(config, mode, batch):
    cfg = replace(config, train=replace(config.train, budget_match=mode),
                  warmup=WarmupConfig(global_batch_size=batch, micro_batch_size=1))
    for vocab in (5, 257):
        steps = warmup_steps(cfg, 257, vocab)
        target = cfg.train.warmup_reference_steps * cfg.train.global_batch_size * cfg.model.seq_len
        cost = batch * cfg.model.seq_len
        if mode == "estimated_flops":
            target *= flops_per_token(cfg.model, 257)
            cost *= flops_per_token(cfg.model, vocab)
        assert 0 <= steps * cost - target < cost
    # Continuous text retains the original batch, budget and schedule.
    assert plan(cfg, "text_continue", 257) == plan(replace(cfg, warmup=None), "text_continue", 257)


@pytest.mark.parametrize("kwargs", [{"lr": -1}, {"weight_decay": -1}, {"global_batch_size": 0},
                                    {"global_batch_size": 2.5}, {"lr": float("nan")}, {"grad_clip": 0}])
def test_invalid_overrides_fail_early(config, kwargs):
    with pytest.raises(ValueError):
        replace(config, warmup=WarmupConfig(**kwargs))


@pytest.mark.parametrize("arm", ["gol_reset", "text_reset"])
def test_source_settings_do_not_leak_into_text(config, corpus, tmp_path, arm):
    cfg = replace(config, warmup=WarmupConfig(global_batch_size=2, micro_batch_size=1,
                                             lr=0.0003, weight_decay=0.02, lr_warmup_steps=0))
    out = tmp_path / arm
    state = run_experiment(cfg, corpus, out, arm, device="cpu")
    warm = torch.load(out / "warmup.pt", weights_only=True)
    final = torch.load(out / "last.pt", weights_only=True)
    assert warm["optimizer"]["param_groups"][0]["weight_decay"] == 0.02
    assert final["optimizer"]["param_groups"][0]["weight_decay"] == cfg.train.weight_decay
    assert all(int(v["step"]) == cfg.train.text_steps for v in final["optimizer"]["state"].values())
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all(r["global_batch_size"] == 2 for r in rows if r["phase"] == "warmup")
    assert all(r["global_batch_size"] == 4 for r in rows if r["phase"] == "text")
    assert [r["learning_rate"] for r in rows if r["phase"] == "warmup" and r["phase_step"] == 1] == [0.0003]
    steps = plan(cfg, arm, TextCorpus(corpus).vocab_size)[0]["steps"]
    assert state["warmup_tokens"] == steps * 2 * cfg.model.seq_len
    assert state["text_tokens"] == cfg.train.text_steps * 4 * cfg.model.seq_len
    assert phase_train_config(cfg, arm, "text") == cfg.train


@pytest.mark.parametrize("where", ["warmup", "boundary", "text"])
def test_resumption_with_stage_overrides(config, corpus, tmp_path, where):
    cfg = replace(config, warmup=WarmupConfig(global_batch_size=2, micro_batch_size=1, lr=0.0003, weight_decay=0))
    warm_steps = plan(cfg, "gol_reset", TextCorpus(corpus).vocab_size)[0]["steps"]
    stop = {"warmup": 1, "boundary": warm_steps, "text": warm_steps + 1}[where]
    full, paused = tmp_path / "full", tmp_path / "paused"
    run_experiment(cfg, corpus, full, "gol_reset", device="cpu")
    run_experiment(cfg, corpus, paused, "gol_reset", device="cpu", stop_after=stop)
    if where == "boundary":
        assert (paused / "warmup.pt").exists()
        assert not (paused / "transfer.json").exists()
    run_experiment(cfg, corpus, paused, "gol_reset", device="cpu", resume=True)
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(paused / "last.pt", weights_only=True)
    assert a["state"]["training_flops_est"] == b["state"]["training_flops_est"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])


def test_print_plan_has_no_training_side_effect(config, corpus, tmp_path):
    cfg = replace(config, warmup=WarmupConfig(global_batch_size=2, micro_batch_size=1))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    command = [sys.executable, "-m", "gol.train", "--config", str(path), "--data", str(corpus),
               "--arm", "gol_reset", "--plan-only"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    stages = json.loads(result.stdout)
    assert stages[0]["training"]["global_batch_size"] == 2
    assert stages[0]["token_exposures"] == stages[0]["steps"] * 2 * cfg.model.seq_len
    assert not list(tmp_path.rglob("*.pt"))


def test_two_rank_stage_batch_transition(config, corpus, tmp_path):
    cfg = replace(config, warmup=WarmupConfig(global_batch_size=2, micro_batch_size=1, lr=0.0003))
    path = tmp_path / "ddp.json"
    path.write_text(json.dumps(cfg.to_dict()))
    single, multi = tmp_path / "single", tmp_path / "multi"
    run_experiment(cfg, corpus, single, "gol_reset", device="cpu")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
         "-m", "gol.train", "--config", str(path), "--data", str(corpus), "--out", str(multi),
         "--arm", "gol_reset", "--device", "cpu"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    a = torch.load(single / "last.pt", weights_only=True)
    b = torch.load(multi / "last.pt", weights_only=True)
    assert a["state"]["warmup_tokens"] == b["state"]["warmup_tokens"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=2e-4, atol=2e-5)
