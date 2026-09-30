"""Aggregate within training seeds before comparing the V2 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate(root):
    records = []
    for evaluation in sorted(root.glob("evaluation-s*")):
        if not evaluation.is_dir():
            continue
        if not (evaluation / "COMPLETE").exists():
            raise RuntimeError(f"Incomplete evaluation: {evaluation}")
        training_seed = int(evaluation.name.split("-s")[-1])
        metrics = json.loads((evaluation / "session-metrics.json").read_text())
        with (evaluation / "episodes.csv").open() as f:
            episodes = list(csv.DictReader(f))
        trajectories = defaultdict(list)
        for e in episodes:
            trajectories[(e["mode"], e["condition"], int(e["seed"]))].append(float(e["return"]))
        for m in metrics:
            change = m["change_episode"]
            values = np.asarray(trajectories[(m["mode"], m["condition"], m["seed"])])
            frozen = trajectories[("frozen", m["condition"], m["seed"])]
            baseline = np.mean(frozen[change - 5 : change])
            records.append(
                dict(
                    training_seed=training_seed,
                    mode=m["mode"],
                    condition=m["condition"],
                    session=m["session"],
                    post_return=float(values[change:].mean()),
                    shared_baseline_regret=float(np.maximum(0, baseline - values[change:]).sum()),
                    recovery_delay=m["recovery_delay"],
                )
            )
    if not records:
        raise RuntimeError("No completed evaluations")
    with (root / "session-results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["mode"], r["condition"], r["training_seed"])].append(r)
    seeds = []
    for (mode, condition, seed), rows in grouped.items():
        seeds.append(
            dict(
                mode=mode,
                condition=condition,
                training_seed=seed,
                post_return=float(np.mean([r["post_return"] for r in rows])),
                regret=float(np.mean([r["shared_baseline_regret"] for r in rows])),
            )
        )
    (root / "training-seed-results.json").write_text(json.dumps(seeds, indent=2))
    lines = [
        "# V2: exploratory multi-seed comparison",
        "",
        "Means are first computed across paired sessions within each training seed. "
        "SD below is variation across independent training seeds, not episodes.",
        "",
        "| Mode | Condition | Seeds | Post-return mean ± SD ↑ | Regret mean ↓ |",
        "|---|---|---:|---:|---:|",
    ]
    for mode, condition in sorted({(r["mode"], r["condition"]) for r in seeds}):
        rows = [r for r in seeds if r["mode"] == mode and r["condition"] == condition]
        returns = [r["post_return"] for r in rows]
        sd = np.std(returns, ddof=1) if len(rows) > 1 else 0.0
        lines.append(
            f"| {mode} | {condition} | {len(rows)} | {np.mean(returns):.3f} ± {sd:.3f} "
            f"| {np.mean([r['regret'] for r in rows]):.2f} |"
        )
    lines += [
        "",
        "## No-change detector activations and rollbacks",
        "",
        "Counts are session-level and reported separately for every training seed.",
        "",
    ]
    for evaluation in sorted(root.glob("evaluation-s*")):
        if not evaluation.is_dir():
            continue
        traces = [
            json.loads(s) for s in (evaluation / "diagnostics.jsonl").read_text().splitlines()
        ]
        for mode in sorted({r["mode"] for r in traces}):
            normal = [r for r in traces if r["mode"] == mode and r["condition"] == "none"]
            alarms = sum(
                any(any(d["semantic"]) or any(d["motor"]) for d in r["diagnostics"]) for r in normal
            )
            rollbacks = sum(sum(r["diagnostics"][-1].get("rollbacks", [0, 0])) for r in normal)
            lines.append(
                f"- {evaluation.name}, {mode}: alarms {alarms}/{len(normal)}, "
                f"rollbacks {rollbacks}."
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "The learned receiver has training-only supervision unavailable to V1/frozen baselines. "
        "Joint V2, unguarded V2 and selective V2 share that exact memory checkpoint. "
        "Motor-only isolates the already-successful motor identification component. "
        "A motor rotation is not a general teammate-policy swap. Three seeds remain exploratory. "
        "Do not tune on these final sessions; retain failures and consult PROTOCOL.md.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    aggregate(parser.parse_args().root)
