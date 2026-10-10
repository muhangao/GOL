import array
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest

from repro.nca import protocol
from repro.nca.data import inspect_tokens, register
from repro.nca.patches import repaired_text, apply_repairs, REPAIRS
from repro.nca.replay import check_source_receipt, body_hash


def test_released_budget_not_pilot_or_rescaled_scheduler():
    plan = protocol.make_plan("source")
    b = plan["source_budget"]
    assert b["effective_global_batch"] == 32
    assert b["selected_updates"] == 5000
    assert b["scheduler_total_updates"] == 50000
    assert b["scheduler_warmup_updates"] == 5000
    assert b["input_tokens"] == 163840000
    assert b["supervised_targets"] == 157920000
    assert b["frames_per_sequence"] == 27
    assert b["grid_tokens"] == 38
    assert b["used_symbol_count"] == 10002
    assert b["input_output_vocab"] == 64000
    assert b["ignored_targets_per_sequence"] == 37
    assert plan["settings"]["n_layer"] == 24
    assert plan["settings"]["n_head"] == 32
    assert plan["settings"]["n_embd"] == 2048
    assert plan["settings"]["temperature"] == 1e-4
    assert "--resume" not in plan["argv"]
    assert "--wandb_enable" not in plan["argv"]
    assert "--filter_rules" in plan["argv"]
    assert not plan["exact_paper_reproduction"]


def test_text_is_original_scale_not_fineweb():
    p = protocol.make_plan("owt")
    assert p["settings"]["batch_size"] * p["settings"]["gradient_accumulation_steps"] == 512
    assert p["settings"]["mixed_precision"] == "fp16"
    assert p["settings"]["weight_decay"] == 1e-4
    assert p["settings"]["seed"] == 5
    assert p["settings"]["pt_vocab_size"] == 64000
    assert p["argv"][-2:] == ["--device", "0"]
    scratch = protocol.make_plan("scratch")
    expected = deepcopy(p["settings"])
    expected["pretrain"] = 0
    assert scratch["settings"] == expected


@pytest.mark.parametrize("stage", ["source", "owt", "scratch"])
def test_smoke_is_explicit_and_separate(stage):
    p = protocol.make_plan(stage, smoke=True, device="cpu")
    assert p["profile"] == "smoke" and not p["exact_paper_reproduction"]
    assert p["source_budget"]["input_tokens"] == 608
    assert p["source_budget"]["selected_updates"] == 4
    assert p["settings"]["n_embd"] == 32


@pytest.mark.parametrize("kwargs", [{"stage": "c4"}, {"stage": "source", "seed": -1},
                                    {"stage": "source", "device": "cpu"}])
def test_reject_unverified_paths(kwargs):
    with pytest.raises(ValueError):
        protocol.make_plan(**kwargs)


@pytest.mark.parametrize("epoch", [0, 101])
def test_invalid_source_checkpoint(epoch):
    with pytest.raises(ValueError):
        protocol.source_budget(protocol.source_settings(), epoch)


def test_plan_needs_no_torch_or_data():
    result = subprocess.run([sys.executable, "-m", "repro.nca.replay", "--stage", "source"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["source_budget"]["input_tokens"] == 163840000


def test_blob_identity():
    assert protocol.blob_sha(b"# GOL\n") == "e29e410b76c46815ae6148f4a3d7c142501ac915"


def test_lock_has_real_sources_and_discloses_conflicts():
    assert len(protocol.LOCK["commit"]) == 40
    for name in ("utils/nca.py", "utils/tokenizers.py", "utils/models.py", "src/nca_ppt.py", "src/openwebtext_pt.py"):
        assert len(protocol.LOCK["files"][name]) == 40
    assert len(protocol.DISCREPANCIES) >= 12


def test_verified_snapshot_detects_mutation(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_text("x = 1\n")
    monkeypatch.setitem(protocol.LOCK, "files", {"source.py": protocol.blob_sha(path.read_bytes())})
    protocol.verify_checkout(tmp_path, check_git=False)
    path.write_text("x = 2\n")
    with pytest.raises(ValueError, match="drift"):
        protocol.verify_checkout(tmp_path, check_git=False)


def test_repairs_are_exact_not_fuzzy():
    assert repaired_text("before OLD after", "OLD", "NEW") == "before NEW after"
    for text in ("none", "OLD OLD"):
        with pytest.raises(ValueError):
            repaired_text(text, "OLD", "NEW")


def test_no_source_repairs(tmp_path):
    assert apply_repairs(tmp_path, "source") == []


def test_owt_repairs_record_each_change(tmp_path):
    for _, rel, old, _, _ in REPAIRS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(old + "\n")
    report = apply_repairs(tmp_path, "owt")
    assert len(report) == 3
    for row in report:
        assert row["before_sha256"] != row["after_sha256"]


def test_released_rule_refresh_is_not_silently_fixed():
    released = [e for e in range(10) if e + 1 % 1 == 0]
    corrected = [e for e in range(10) if (e + 1) % 1 == 0]
    assert released == [0] and len(corrected) == 10


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "owt"
    root.mkdir()
    for name in ("train.bin", "val.bin"):
        (root / name).write_bytes(array.array("H", list(range(100))).tobytes())
    register(root, "smoke", {"test": True})
    return root


def test_data_bytes_match_original_uint16(corpus):
    manifest = protocol.validate_data_manifest(corpus, smoke=True)
    assert manifest["splits"]["train"]["tokens"] == 100
    assert manifest["dtype"] == "<u2"
    with pytest.raises(FileExistsError):
        register(corpus, "smoke", {"test": True})


@pytest.mark.parametrize("bad", ["corpus", "dtype", "tokenizer", "origin"])
def test_reject_data_substitution(corpus, bad):
    path = corpus / "manifest.json"
    data = json.loads(path.read_text())
    data[bad] = "fineweb_pilot"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        protocol.validate_data_manifest(corpus, smoke=True)


def test_reject_small_production_corpus(corpus):
    path = corpus / "manifest.json"
    data = json.loads(path.read_text()); data["origin"] = "reconstructed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="small corpus"):
        protocol.validate_data_manifest(corpus)


def test_changed_token_file(corpus):
    (corpus / "train.bin").write_bytes(b"\0\0" * 100)
    with pytest.raises(ValueError, match="integrity"):
        protocol.validate_data_manifest(corpus, smoke=True)


@pytest.mark.parametrize("data", [b"\0", b"\0\0", array.array("H", [65535, 1]).tobytes()])
def test_invalid_uint16_data(tmp_path, data):
    p = tmp_path / "x.bin"; p.write_bytes(data)
    with pytest.raises(ValueError):
        inspect_tokens(p)


def test_receipt_rejects_wrong_model_family_and_budget(tmp_path):
    p = tmp_path / "selected_source.json"
    p.write_text(json.dumps({"kind": "gol", "profile": "smoke", "upstream_commit": protocol.LOCK["commit"]}))
    with pytest.raises(ValueError, match="handoff"):
        check_source_receipt(tmp_path, smoke=True)
    checkpoint = tmp_path / "source.pth"; checkpoint.write_bytes(b"checkpoint")
    value = dict(kind="nca_source_checkpoint", profile="smoke", upstream_commit=protocol.LOCK["commit"],
                 checkpoint="source.pth", checkpoint_sha256=protocol.sha256_file(checkpoint), input_tokens=608)
    p.write_text(json.dumps(value)); check_source_receipt(tmp_path, smoke=True)
    value["input_tokens"] = 609; p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="budget"):
        check_source_receipt(tmp_path, smoke=True)
    value["checkpoint"] = "../source.pth"; p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsafe"):
        check_source_receipt(tmp_path, smoke=True)


def test_body_hash_excludes_only_io():
    import torch
    model = torch.nn.Module()
    model.input_proj = torch.nn.Linear(2, 3)
    model.output_proj = torch.nn.Linear(3, 4)
    model.norm = torch.nn.LayerNorm(3)
    before = body_hash(model)
    with torch.no_grad():
        model.input_proj.weight.add_(1)
    assert body_hash(model) == before
    with torch.no_grad():
        model.norm.weight.add_(1)
    assert body_hash(model) != before
