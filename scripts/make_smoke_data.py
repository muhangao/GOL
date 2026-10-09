"""Generate original toy text to exercise the pipeline, not to evaluate the hypothesis."""
import argparse
from pathlib import Path


def make_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split, offset in (("warmup", 0), ("train", 1000), ("valid", 2000)):
        path = root / f"{split}.txt"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        lines = [f"Record {i + offset}: A small model reads a sequence and predicts the next symbol. "
                 f"The experiment counts tokens and measures loss on a separate document {i + offset}."
                 for i in range(200)]
        path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/smoke_raw")
    make_data(parser.parse_args().out)
