"""Export native held-out loss curves without mixing pilots, profiles or corpora."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(runs: Path, output: Path, *, include_smoke=False):
    records, curves = [], []
    signature = None
    seen = set()
    for done in sorted(runs.rglob("complete.json")):
        root = done.parent
        completed = json.loads(done.read_text())
        if completed.get("stage") not in {"owt", "scratch"}:
            continue
        if completed["profile"] == "smoke" and not include_smoke:
            continue
        plan = json.loads((root / "plan.json").read_text())
        prov = json.loads((root / "provenance.json").read_text())
        params = {k: v for k, v in plan["settings"].items() if k not in {"seed", "pretrain"}}
        current = {"profile": plan["profile"], "upstream": plan["upstream_commit"], "settings": params,
                   "data_hashes": {k: v["sha256"] for k, v in prov["data"]["splits"].items()},
                   "runtime": prov["runtime"], "wrapper": prov["wrapper_files"]}
        # Allocation IDs differ without changing the numerical environment.
        current["runtime"] = {k: v for k, v in current["runtime"].items() if k != "selected_environment"}
        if signature is None:
            signature = current
        elif current != signature:
            raise ValueError("Refusing to aggregate different data, settings, code, profile or runtime")
        seed, arm = plan["settings"]["seed"], plan["stage"]
        if (arm, seed) in seen:
            raise ValueError(f"Duplicate {arm}/seed {seed}")
        seen.add((arm, seed))
        source_tokens = 0 if arm == "scratch" else prov["source_receipt"]["input_tokens"]
        records.append(dict(arm=arm, seed=seed, loss=completed["final_validation_loss"],
                            text_input_tokens=completed["input_tokens"], source_input_tokens=source_tokens,
                            total_input_tokens=completed["input_tokens"] + source_tokens,
                            successful_optimizer_steps=completed["successful_optimizer_steps"]))
        for line in (root / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "validation":
                curves.append(dict(arm=arm, seed=seed, loss=event["loss"],
                                   text_input_tokens=event["input_tokens"],
                                   total_input_tokens=event["input_tokens"] + source_tokens,
                                   successful_optimizer_steps=event["successful_optimizer_steps"]))
    if not records:
        raise ValueError("No completed matching NCA/OWT replay runs; smoke is excluded by default")
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("final_losses.csv", records), ("loss_curves.csv", curves)):
        with (output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {}
    for arm in {r["arm"] for r in records}:
        values = [r["loss"] for r in records if r["arm"] == arm]
        summary[arm] = dict(n=len(values), mean=statistics.mean(values),
                            sample_sd=statistics.stdev(values) if len(values) > 1 else None)
    (output / "summary.json").write_text(json.dumps(dict(
        summary=summary, metric="native validation cross-entropy (mean of batch means)",
        budget="input-token exposures, NOT hardware FLOPs", exact_paper_reproduction=False), indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--include-smoke", action="store_true")
    a = p.parse_args()
    print(json.dumps(summarize(a.runs, a.out, include_smoke=a.include_smoke), indent=2))


if __name__ == "__main__":
    main()
