from dataclasses import replace
import numpy as np
import pytest
import torch
from gol import life
from gol.config import Config, ModelConfig, LifeConfig, flops_per_token, warmup_steps
from gol.model import build_model, transfer_body, fingerprint


def test_still_life():
    board = np.zeros((6, 6), bool)
    board[2:4, 2:4] = True
    for boundary in ("toroidal", "dead"):
        np.testing.assert_array_equal(life.step(board, boundary), board)


def test_blinker():
    board = np.zeros((5, 5), bool)
    board[2, 1:4] = True
    expected = np.zeros_like(board)
    expected[1:4, 2] = True
    np.testing.assert_array_equal(life.step(board), expected)
    np.testing.assert_array_equal(life.step(expected), board)


def test_glider_and_wrapping():
    board = np.zeros((8, 8), bool)
    board[0, 1] = board[1, 2] = True
    board[2, :3] = True
    moved = board.copy()
    for _ in range(4):
        moved = life.step(moved)
    np.testing.assert_array_equal(moved, np.roll(board, (1, 1), axis=(0, 1)))
    edge = np.zeros((5, 5), bool)
    edge[0, [0, 1, 4]] = True
    assert not np.array_equal(life.step(edge, "dead"), life.step(edge, "toroidal"))


def test_serialization(config):
    a = life.sample(5, 10, 65, config.life)
    np.testing.assert_array_equal(a, life.sample(5, 10, 65, config.life))
    assert not np.array_equal(a, life.sample(6, 10, 65, config.life))
    assert len(a) == 65 and set(a) <= set(range(life.VOCAB_SIZE))
    assert a[0] == life.BOS and a[17] == life.FRAME
    initial = a[1:17].reshape(4, 4)
    np.testing.assert_array_equal(a[18:34].reshape(4, 4), life.step(initial))


def test_independent_initialization_and_transfer(config):
    a = build_model(config.model, 5, 42, 43)
    b = build_model(config.model, 257, 42, 43)
    assert fingerprint(a) == fingerprint(b)
    with torch.no_grad():
        a.blocks[0].qkv.weight.add_(0.1)
        a.norm.weight.fill_(1.5)
    moved = transfer_body(a, 257, 42, 44)
    fresh = build_model(config.model, 257, 42, 44)
    assert fingerprint(a) == fingerprint(moved)
    assert fingerprint(moved, True) == fingerprint(fresh, True)
    assert moved.embedding.weight.data_ptr() == moved.lm_head.weight.data_ptr()
    assert moved.embedding.weight.shape == (257, 32)


def test_untied_interface(config):
    cfg = replace(config.model, tied_embeddings=False)
    model = transfer_body(build_model(cfg, 5, 1, 2), 257, 1, 3)
    assert model.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert fingerprint(model, True) == fingerprint(build_model(cfg, 257, 1, 3), True)


def test_causality_and_gradients(config):
    model = build_model(config.model, 257, 0, 1)
    tokens = torch.randint(0, 257, (2, 64))
    changed = tokens.clone()
    changed[:, 32:] = (changed[:, 32:] + 1) % 257
    before, after = model(tokens), model(changed)
    torch.testing.assert_close(before[:, :32], after[:, :32], rtol=0, atol=0)
    loss = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_bfloat16_rope_dtype(config):
    model = build_model(config.model, 257, 0, 1)
    tokens = torch.randint(0, 257, (2, 64))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(tokens, tokens)
    loss.backward()
    assert torch.isfinite(loss)


def test_budget_modes(config):
    gol_cost = flops_per_token(config.model, 5)
    text_cost = flops_per_token(config.model, 50257)
    steps = warmup_steps(config, 50257, 5)
    target = config.train.warmup_reference_steps * text_cost
    assert 0 <= steps * gol_cost - target < gol_cost
    matched = replace(config, train=replace(config.train, budget_match="tokens"))
    assert warmup_steps(matched, 50257, 5) == config.train.warmup_reference_steps
    assert warmup_steps(config, 50257, 50257) == config.train.warmup_reference_steps


def test_invalid_config(config):
    with pytest.raises(ValueError):
        ModelConfig(dim=31, heads=4)
    with pytest.raises(ValueError):
        Config.from_dict({"unknown": True})
    with pytest.raises(ValueError):
        replace(config, life=LifeConfig(board_size=12))
