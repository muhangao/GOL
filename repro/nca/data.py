"""Prepare or register OWT assets without claiming to know the authors' split.

The released preprocess.py prepares a math TEST split, not the OWT train/val
assets. A reconstructed OWT corpus must therefore be explicitly labeled as such.
"""
from __future__ import annotations
import argparse
import array
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from .protocol import sha256_file, write_json


def inspect_tokens(path: Path) -> dict:
    if sys.byteorder != "little":
        raise ValueError("This replay requires a little-endian host, like the released numpy.uint16 loader")
    size = path.stat().st_size
    if size % 2 or size < 4:
        raise ValueError(f"Invalid uint16 file size: {path}")
    h = hashlib.sha256()
    maximum = 0
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
            values = array.array("H")
            values.frombytes(data)
            maximum = max(maximum, max(values))
    if maximum >= 50257:
        raise ValueError(f"Non-GPT-2 token ID in {path}: {maximum}")
    return dict(file=path.name, tokens=size // 2, sha256=h.hexdigest(), maximum_id=maximum)


def register(directory: Path, origin: str, provenance: dict) -> dict:
    directory = Path(directory).resolve(strict=True)
    if (directory / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing data manifest")
    if origin not in {"declared_author_assets", "reconstructed", "smoke"} or not provenance:
        raise ValueError("Specify a data origin and nonempty provenance")
    manifest = dict(schema_version=1, corpus="openwebtext", dtype="<u2", tokenizer="tiktoken:gpt2",
                    origin=origin, provenance=provenance,
                    note="The author's exact OWT split/hashes were not provided in the release. Origin is a recorded declaration, not independent authentication.",
                    splits={"train": inspect_tokens(directory / "train.bin"),
                            "validation": inspect_tokens(directory / "val.bin")})
    write_json(directory / "manifest.json", manifest)
    return manifest


def prepare(directory: Path, revision: str, validation_fraction: float, split_seed: int, num_proc: int):
    if not 0 < validation_fraction < 0.1 or split_seed < 0 or num_proc < 1:
        raise ValueError("Invalid split policy or worker count")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin --revision to a full immutable Hugging Face dataset commit SHA")
    if directory.exists():
        raise FileExistsError(directory)
    import datasets
    import tiktoken
    import numpy as np
    enc = tiktoken.get_encoding("gpt2")
    source = datasets.load_dataset("Skylion007/openwebtext", revision=revision, split="train")
    split = source.train_test_split(test_size=validation_fraction, seed=split_seed, shuffle=True)
    directory.mkdir(parents=True)

    def encode(example):
        return {"ids": enc.encode_ordinary(example["text"]) + [enc.eot_token]}

    fingerprints = {}
    for name, key in (("train", "train"), ("val", "test")):
        encoded = split[key].map(encode, remove_columns=split[key].column_names,
                                 num_proc=num_proc, desc=f"GPT-2 encode {name}")
        fingerprints[name] = {"documents": len(split[key]), "raw": split[key]._fingerprint,
                              "tokenized": encoded._fingerprint}
        # No preallocated zero tail, source truncation, resampling or FineWeb substitution.
        with (directory / f"{name}.bin").open("wb") as f:
            for row in encoded:
                f.write(np.asarray(row["ids"], dtype="<u2").tobytes())
    return register(directory, "reconstructed", dict(
        dataset="Skylion007/openwebtext", revision=revision,
        split_policy={"validation_fraction": validation_fraction, "seed": split_seed, "shuffle": True},
        encoding="encode_ordinary(text) followed by one GPT-2 EOT per document",
        tiktoken_version=importlib.metadata.version("tiktoken"),
        datasets_version=importlib.metadata.version("datasets"), fingerprints=fingerprints))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("register", help="Register existing, trusted author or reconstructed binary files")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--origin", choices=("declared_author_assets", "reconstructed"), required=True)
    p.add_argument("--provenance-json", type=Path, required=True)
    p = commands.add_parser("prepare", help="Explicitly reconstruct OWT; not the undisclosed author split")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--validation-fraction", required=True, type=float)
    p.add_argument("--split-seed", required=True, type=int)
    p.add_argument("--num-proc", type=int, default=16)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.data_dir, args.origin, json.loads(args.provenance_json.read_text()))
    else:
        result = prepare(args.out, args.revision, args.validation_fraction, args.split_seed, args.num_proc)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
