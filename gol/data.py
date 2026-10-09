"""Streaming corpus preparation and deterministic, memory-mapped token sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

DTYPE = np.dtype("<u4")
SPLITS = ("valid", "warmup", "train")  # Evaluation documents take priority during deduplication.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def documents(path: Path):
    """One document per nonempty TXT line or JSONL object with a string `text`."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if path.suffix == ".jsonl":
                try:
                    text = json.loads(line)["text"]
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: expected an object with a text field") from exc
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{line_number}: text must be a string")
            else:
                text = line
            if text.strip():
                yield text.strip()


def prepare(output, inputs, tokenizer="byte", eos_token=None, revision="main"):
    output = Path(output)
    inputs = {name: Path(inputs[name]).resolve(strict=True) for name in SPLITS}
    if len(set(inputs.values())) != len(SPLITS):
        raise ValueError("Provide different files for warmup, train and valid")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use a new output directory")
    if tokenizer == "byte":
        encode = lambda text: list(text.encode("utf-8"))
        eos_id, vocab_size = 256, 257
        tokenizer_info = {"kind": "utf8_byte", "eos_id": eos_id, "vocab_size": vocab_size}
        tokenizer_json = None
    else:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install the tokenize extra for a text tokenizer: pip install -e '.[tokenize]'") from exc
        if not eos_token:
            raise ValueError("Supply --eos-token explicitly for non-byte tokenizers")
        local = Path(tokenizer)
        if local.is_dir():
            local = local / "tokenizer.json"
        if local.is_file():
            tok = Tokenizer.from_file(str(local))
            origin = str(local.resolve())
        else:
            from huggingface_hub import hf_hub_download
            local = Path(hf_hub_download(tokenizer, "tokenizer.json", revision=revision))
            tok = Tokenizer.from_file(str(local))
            origin = tokenizer
        tok.no_padding()
        tok.no_truncation()
        eos_id = tok.token_to_id(eos_token)
        if eos_id is None:
            raise ValueError(f"EOS token {eos_token!r} is absent from the vocabulary")
        # Do not silently insert a model-specific BOS/EOS template.
        encode = lambda text: tok.encode(text, add_special_tokens=False).ids
        vocab_size = max(tok.get_vocab().values()) + 1
        tokenizer_json = tok.to_str()
        tokenizer_info = {"kind": "tokenizers_json", "origin": origin, "revision": revision,
                          "eos_token": eos_token, "eos_id": eos_id, "vocab_size": vocab_size,
                          "sha256": hashlib.sha256(tokenizer_json.encode()).hexdigest()}
    if vocab_size > np.iinfo(DTYPE).max:
        raise ValueError("Vocabulary does not fit uint32")
    output.mkdir(parents=True)
    if tokenizer_json is not None:
        (output / "tokenizer.json").write_text(tokenizer_json, encoding="utf-8")
    manifest = {"schema_version": 1, "dtype": DTYPE.str, "tokenizer": tokenizer_info,
                "deduplication": "exact stripped document SHA256; priority valid, warmup, train",
                "splits": {}}
    database = output / "dedup.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY, split TEXT NOT NULL)")
        for split in SPLITS:
            path = inputs[split]
            count = kept = dropped = 0
            digest = hashlib.sha256()
            with (output / f"{split}.bin").open("wb") as handle:
                for text in documents(path):
                    key = hashlib.sha256(text.encode("utf-8")).digest()
                    cursor = db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (key, split))
                    if not cursor.rowcount:
                        dropped += 1
                        continue
                    ids = encode(text) + [eos_id]
                    if min(ids) < 0 or max(ids) >= vocab_size:
                        raise ValueError("Tokenizer emitted an out-of-range ID")
                    data = np.asarray(ids, dtype=DTYPE).tobytes()
                    handle.write(data)
                    digest.update(data)
                    count += len(ids)
                    kept += 1
                    if kept % 10000 == 0:
                        db.commit()
            db.commit()
            if count < 2:
                raise ValueError(f"Split {split} is empty after deduplication; choose different source data")
            manifest["splits"][split] = {"file": f"{split}.bin", "tokens": count,
                                           "documents": kept, "duplicates_dropped": dropped,
                                           "sha256": digest.hexdigest(), "source": str(path),
                                           "source_sha256": sha256_file(path)}
    database.unlink()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class TextCorpus:
    def __init__(self, directory, verify=False):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        self.fingerprint = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != 1 or self.manifest.get("dtype") != DTYPE.str:
            raise ValueError("Unsupported corpus manifest")
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        self.arrays = {}
        for split in SPLITS:
            meta = self.manifest["splits"][split]
            path = self.directory / meta["file"]
            if path.stat().st_size != meta["tokens"] * DTYPE.itemsize:
                raise ValueError(f"Token count/file size mismatch in {split}")
            if verify and sha256_file(path) != meta["sha256"]:
                raise ValueError(f"Checksum mismatch in {split}")
            self.arrays[split] = np.memmap(path, mode="r", dtype=DTYPE)

    def sample(self, split: str, index: int, seed: int, length: int) -> np.ndarray:
        tokens = self.arrays[split]
        if length > len(tokens):
            raise ValueError(f"{split} has {len(tokens)} tokens, but a sample needs {length}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        offset = int(rng.integers(0, len(tokens) - length + 1))
        result = tokens[offset:offset + length].astype(np.int64)
        if result.min() < 0 or result.max() >= self.vocab_size:
            raise ValueError(f"Out-of-vocabulary token in {split}")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--valid", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", default="byte", help="byte, a tokenizer.json path, or a Hub repository")
    parser.add_argument("--eos-token", help="Required for a JSON/Hub tokenizer; must already exist")
    parser.add_argument("--revision", default="main", help="Pin a Hub commit for immutable provenance")
    args = parser.parse_args()
    result = prepare(args.out, {k: getattr(args, k) for k in SPLITS}, args.tokenizer, args.eos_token, args.revision)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
