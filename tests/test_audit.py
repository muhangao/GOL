from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from gol.audit import (BASELINES, Totals, audit, examples, features, fit_baselines,
                       layout, load_source, rule_reference_losses, score_model)
from gol.config import Config, LifeConfig
from gol.data import sha256_file
from gol.life import BOS, EOS, FRAME, sample
from gol.model import build_model


@pytest.fixture
def legacy_checkpoint(config, tmp_path):
    model = build_model(config.model, 5, 7, 8)
    path = tmp_path / "old_run" / "warmup.pt"
    path.parent.mkdir()
    # Legacy schema: no warmup override, no resolved per-stage training fields.
    torch.save({"metadata": {"schema_version": 1, "arm": "gol_reset", "seed": 7,
                            "config": config.to_dict(), "code_sha256": "legacy-training-code",
                            "stages": [{"name": "warmup", "source": "gol", "vocab": 5, "steps": 4}]},
                "state": {"phase_index": 0, "step": 4}, "model": model.state_dict(),
                "optimizer": {"state": {}, "param_groups": []}}, path)
    return path


def test_pilot_packing_counts():
    cfg = Config()
    r = layout(cfg.life, cfg.model.seq_len)
    assert (r.initial.sum(), r.evolved.sum(), (~r.cell).sum()) == (288, 727, 9)
    # The second BOS is absolute position 872, followed by another initial board.
    assert r.delimiter[871] == BOS and not r.cell[871]
    assert r.initial[872:1016].all()
    assert r.delimiter[1016] == FRAME and r.evolved[1017:].all()


@pytest.mark.parametrize("side,frames", [(3, 2), (4, 3), (12, 6)])
def test_layout_matches_serialization(side, frames):
    life = LifeConfig(board_size=side, frames=frames)
    length = 3 * (2 + frames * (side * side + 1)) + 7
    tokens = sample(2, 42, length + 1, life)
    roles = layout(life, length)
    np.testing.assert_array_equal(tokens[1:] <= 1, roles.cell)
    np.testing.assert_array_equal(tokens[1:][~roles.cell], roles.delimiter[~roles.cell])
    frame = -1
    for j, token in enumerate(tokens):
        if token == BOS:
            frame = 0
        elif token == FRAME:
            frame += 1
        elif token != EOS and j > 0:
            assert roles.frame[j - 1] == frame
            assert roles.initial[j - 1] == (frame == 0)


@pytest.mark.parametrize("boundary", ["dead", "toroidal"])
def test_known_rule_reference_is_causal_and_correct(config, boundary):
    cfg = replace(config, life=replace(config.life, boundary=boundary))
    roles = layout(cfg.life, cfg.model.seq_len)
    tokens = next(examples(cfg, 10, 5, 4321, 5))
    losses = rule_reference_losses(tokens, cfg, roles)
    assert np.all(losses[:, ~roles.initial] == 0)
    assert np.all(losses[:, roles.initial] > 0)


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_features_never_read_target_or_future(config, name):
    roles = layout(config.life, config.model.seq_len)
    tokens = next(examples(config, 0, 2, 4321, 2))
    original = features(tokens, roles, name)
    for position in range(1, config.model.seq_len + 1):
        changed = tokens.copy()
        changed[:, position:] = (changed[:, position:] + 1) % 5
        np.testing.assert_array_equal(features(changed, roles, name)[:, position - 1], original[:, position - 1])


def test_totals_reconstruct_whole_loss_and_handle_empty_group(config):
    roles = layout(config.life, config.model.seq_len)
    tokens = next(examples(config, 0, 3, 12, 3))
    targets = tokens[:, 1:]
    losses = np.arange(targets.size).reshape(targets.shape) / 100
    t = Totals()
    t.add(losses[:2], targets[:2], roles)
    t.add(losses[2:], targets[2:], roles)
    result = t.result()
    total = sum(result[g]["nll_sum"] for g in ("initial_cells", "evolved_cells", "delimiters"))
    assert total == pytest.approx(result["all"]["nll_sum"])
    assert total / targets.size == pytest.approx(losses.mean())
    empty = Totals()
    empty.add(np.zeros((1, 4)), np.zeros((1, 4), dtype=int), layout(config.life, 4))
    assert empty.result()["evolved_alive"]["loss"] is None


def test_model_loss_alignment(config):
    model = build_model(config.model, 5, 7, 8)
    tokens = next(examples(config, 7, 3, 12, 3))
    values = torch.from_numpy(tokens)
    with torch.no_grad():
        expected = model(values[:, :-1], values[:, 1:]).item()
    groups = score_model(model, config, 7, 3, 12, 2, torch.device("cpu"), "float32")
    assert groups["all"]["loss"] == pytest.approx(expected, abs=3e-7)
    assert all(p.grad is None for p in model.parameters())


def test_fit_uses_only_fit_examples_and_is_batch_invariant(config):
    first = fit_baselines(config, 13, 891, 3, 0.5)
    second = fit_baselines(config, 13, 891, 8, 0.5)
    for name in BASELINES:
        np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_allclose(first[name].sum(1), 1)


def test_old_checkpoint_audit_is_readonly_and_deterministic(legacy_checkpoint, tmp_path):
    before = sha256_file(legacy_checkpoint)
    original, cfg = load_source(legacy_checkpoint)
    assert "warmup" not in original["metadata"]["config"]
    report = audit([legacy_checkpoint], tmp_path / "audit", fit_sequences=13, eval_sequences=11, batch_size=3, device="cpu")
    assert sha256_file(legacy_checkpoint) == before
    assert report["fit_index_range"] == [0, 13] and report["eval_index_range"] == [13, 24]
    records = report["records"]
    assert len(records) == 6 and records[-1]["kind"] == "checkpoint"
    assert records[-1]["training_code_sha256"] == "legacy-training-code"
    assert all(r["groups"]["all"]["tokens"] == 11 * cfg.model.seq_len for r in records)
    assert (tmp_path / "audit/losses.csv").exists()
    serialized = json.loads((tmp_path / "audit/report.json").read_text())
    assert serialized == report
    again = audit([legacy_checkpoint], tmp_path / "audit2", fit_sequences=13, eval_sequences=11, batch_size=3, device="cpu")
    assert records == again["records"]
    with pytest.raises(FileExistsError):
        audit([legacy_checkpoint], tmp_path / "audit", fit_sequences=13, eval_sequences=11, device="cpu")


@pytest.mark.parametrize("mutation", ["phase", "arm", "vocab"])
def test_rejects_non_source_checkpoints(legacy_checkpoint, mutation):
    payload = torch.load(legacy_checkpoint, weights_only=True)
    if mutation == "phase":
        payload["state"]["phase_index"] = 1
    elif mutation == "arm":
        payload["metadata"]["arm"] = "text_reset"
    else:
        payload["model"]["embedding.weight"] = torch.zeros(257, 32)
    torch.save(payload, legacy_checkpoint)
    with pytest.raises(ValueError):
        load_source(legacy_checkpoint)


def test_rejects_seed_overlap_and_generator_mismatch(legacy_checkpoint, config, tmp_path):
    with pytest.raises(ValueError, match="collides"):
        audit([legacy_checkpoint], tmp_path / "audit", seed=config.train.data_seed + 1)
    other = replace(config, life=replace(config.life, boundary="dead"))
    with pytest.raises(ValueError, match="generator"):
        audit([legacy_checkpoint], tmp_path / "audit", config=other)
    with pytest.raises(ValueError, match="Duplicate"):
        audit([legacy_checkpoint, legacy_checkpoint], tmp_path / "audit")


def test_baselines_only_and_input_validation(config, tmp_path):
    report = audit([], tmp_path / "baseline", config, fit_sequences=7, eval_sequences=5, device="cpu")
    assert len(report["records"]) == 5
    assert not any(r["kind"] == "checkpoint" for r in report["records"])
    with pytest.raises(ValueError, match="Supply"):
        audit([], tmp_path / "missing")
    with pytest.raises(ValueError):
        audit([], tmp_path / "zero", config, fit_sequences=0)
    with pytest.raises(ValueError):
        audit([], tmp_path / "alpha", config, alpha=float("nan"))
