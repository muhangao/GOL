"""Split FineWeb-Edu parquet shards into disjoint warmup/train/valid JSONL files."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def rows(path):
    for group in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
        yield from group.column("text").to_pylist()


def write(handle, text):
    handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", required=True, type=Path, help="Every document goes to train")
    parser.add_argument("--heldout-shard", required=True, type=Path, help="Split into valid, then warmup")
    parser.add_argument("--valid-docs", type=int, default=10000)
    parser.add_argument("--warmup-docs", type=int, default=100000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "train.jsonl").open("w") as train:
        for text in rows(args.train_shard):
            write(train, text)
    with (args.out / "valid.jsonl").open("w") as valid, (args.out / "warmup.jsonl").open("w") as warmup:
        for i, text in enumerate(rows(args.heldout_shard)):
            if i < args.valid_docs:
                write(valid, text)
            elif i < args.valid_docs + args.warmup_docs:
                write(warmup, text)
            else:
                break


if __name__ == "__main__":
    main()
