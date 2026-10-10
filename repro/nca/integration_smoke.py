"""Execute the actual pinned drivers at tiny sizes in isolated child processes.

No production corpus, GPU benchmark, high-complexity band result, or transfer
benefit is claimed. This checks actual upstream imports, generation, target
masking, source optimization, checkpoint selection, handoff and OWT training.
"""
from __future__ import annotations
import argparse
import array
import json
from pathlib import Path
import subprocess
import sys
from .data import register
from .protocol import LOCK


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.out / "data"
    data.mkdir()
    # Deliberately exact block multiples: exercise the disclosed final-target repair.
    for filename, length in (("train.bin", 9 * 76), ("val.bin", 5 * 76)):
        (data / filename).write_bytes(array.array("H", [i % 17 for i in range(length)]).tobytes())
    register(data, "smoke", {"description": "generated integer tokens, not natural language or author assets"})
    for stage in ("source", "owt", "scratch"):
        command = [sys.executable, "-m", "repro.nca.replay", "--stage", stage,
                   "--upstream", str(args.upstream.resolve()), "--out", str((args.out / stage).resolve()),
                   "--smoke", "--device", "cpu", "--execute"]
        if stage != "source":
            command += ["--data", str(data.resolve())]
        if stage == "owt":
            command += ["--source-run", str((args.out / "source").resolve())]
        subprocess.run(command, check=True, timeout=240)
        completed = json.loads((args.out / stage / "complete.json").read_text())
        assert completed["profile"] == "smoke" and not completed["exact_paper_reproduction"]
        assert completed["successful_optimizer_steps"] > 0
    source = json.loads((args.out / "source/selected_source.json").read_text())
    assert source["input_tokens"] == 608 and source["native_iterations"] == 4
    for stage in ("source", "owt", "scratch"):
        structure = json.loads((args.out / stage / "model_and_optimizer.json").read_text())
        assert not structure["tied"]
        assert structure["input_shape"][1] == 32
    print("Pinned upstream source + NCA-to-OWT + scratch CPU smoke completed.")


if __name__ == "__main__":
    main()
