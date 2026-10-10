"""Plot held-out text loss against downstream tokens and total estimated compute."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def collect(root, include_incomplete=False):
    groups = defaultdict(list)
    reference = None
    seen = set()
    for path in sorted(Path(root).rglob("metrics.jsonl")):
        run = path.parent
        if not include_incomplete and not (run / "complete.json").exists():
            continue
        meta = json.loads((run / "metadata.json").read_text())
        signature = (meta["config"], meta["manifest_sha256"], meta["code_sha256"],
                     meta["world_size"], meta["device_type"], meta["torch"])
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("Refusing to aggregate runs with different config, corpus, code or runtime")
        key = (meta["arm"], meta["seed"])
        if key in seen:
            raise ValueError(f"Duplicate arm/seed in {root}: {key}")
        seen.add(key)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["text_val_loss"] is not None]
        if rows:
            groups[meta["arm"]].append((meta["seed"], rows))
    if not groups:
        raise ValueError("No completed runs with text validation loss found")
    return groups


def plot(root, output, include_incomplete=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = collect(root, include_incomplete)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for x_key, label, filename in (
        ("text_tokens", "Tokens in the common downstream text stage", "loss_vs_text_tokens.png"),
        ("training_flops_est", "Total estimated training matmul FLOPs (including warmup)", "loss_vs_total_flops.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm, runs in sorted(groups.items()):
            maps = [{r[x_key]: r["text_val_loss"] for r in rows
                     if r["phase"] == "text" or (x_key == "training_flops_est" and arm == "text_continue")}
                    for _, rows in runs]
            # Compare only checkpoints observed in every seed; never extrapolate.
            common = sorted(set.intersection(*(set(m) for m in maps)))
            if not common:
                raise ValueError(f"No common evaluation checkpoints across seeds for {arm}")
            values = np.array([[m[x] for x in common] for m in maps])
            mean = values.mean(0)
            ax.plot(common, mean, label=f"{arm} (n={len(runs)})")
            if len(runs) > 1:
                std = values.std(0, ddof=1)
                ax.fill_between(common, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel(label)
        ax.set_ylabel("Held-out text cross-entropy (nats/token)")
        ax.set_title("Mean loss; shading is seed standard deviation, not a confidence interval")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
    with (output / "final_losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "phase_step", "text_val_loss", "text_tokens", "warmup_tokens", "training_flops_est"])
        for arm, runs in sorted(groups.items()):
            for seed, rows in runs:
                last = rows[-1]
                writer.writerow([arm, seed] + [last[k] for k in ("phase_step", "text_val_loss", "text_tokens", "warmup_tokens", "training_flops_est")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args()
    plot(args.runs, args.out, args.include_incomplete)


if __name__ == "__main__":
    main()
