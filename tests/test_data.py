import json
import numpy as np
import pytest
from gol.data import TextCorpus, prepare


def test_corpus_reproducible_sampling(corpus):
    data = TextCorpus(corpus, verify=True)
    a = data.sample("train", 42, 7, 65)
    np.testing.assert_array_equal(a, data.sample("train", 42, 7, 65))
    assert a.shape == (65,) and a.dtype == np.int64
    assert a.min() >= 0 and a.max() < data.vocab_size
    with pytest.raises(ValueError):
        data.sample("valid", 0, 0, 10000000)


def test_document_deduplication(tmp_path):
    paths = {}
    for split in ("warmup", "train", "valid"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"text": "shared document"}) + "\n" +
                        json.dumps({"text": f"unique {split} document"}) + "\n")
        paths[split] = path
    meta = prepare(tmp_path / "tokens", paths)
    assert meta["splits"]["valid"]["duplicates_dropped"] == 0
    for split in ("warmup", "train"):
        assert meta["splits"][split]["duplicates_dropped"] == 1
    assert meta["tokenizer"]["eos_id"] == 256


def test_corrupt_size_and_hash(corpus):
    path = corpus / "train.bin"
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="size mismatch"):
        TextCorpus(corpus)
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="Checksum"):
        TextCorpus(corpus, verify=True)


def test_refuse_overwrite_and_same_source(corpus, tmp_path):
    paths = {split: tmp_path / f"{split}.txt" for split in ("train", "valid", "warmup")}
    with pytest.raises(FileExistsError):
        prepare(corpus, paths)
    with pytest.raises(ValueError, match="different files"):
        prepare(tmp_path / "new", {split: paths["train"] for split in paths})


def test_local_json_tokenizer(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer_path))
    paths = {}
    for split in ("warmup", "train", "valid"):
        path = tmp_path / f"{split}.txt"
        path.write_text(f"hello {split}\n")
        paths[split] = path
    out = tmp_path / "prepared"
    meta = prepare(out, paths, str(tokenizer_path), "<eos>")
    assert meta["tokenizer"]["vocab_size"] == 3
    assert (out / "tokenizer.json").exists()
    np.testing.assert_array_equal(TextCorpus(out).arrays["train"], [2, 0, 1])
