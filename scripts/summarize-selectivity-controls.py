"""Validate complete paired evaluations and analyze the preregistered endpoint."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

from selective_marl.adapters_v2 import V2_MODES


def paired_interval(values, confidence=0.95):
    x = np.asarray(values, float)
    mean = float(x.mean())
    if len(x) < 2:
        return dict(mean=mean, ci95=None, lower_one_sided95=None, p_one_sided=None)
    se = float(stats.sem(x))
    half = float(stats.t.ppf((1 + confidence) / 2, len(x) - 1) * se)
    p = float(stats.t.sf(mean / se, len(x) - 1)) if se else (0.0 if mean > 0 else 1.0)
    return dict(
        mean=mean,
        ci95=[mean - half, mean + half],
        lower_one_sided95=mean - float(stats.t.ppf(confidence, len(x) - 1)) * se,
        p_one_sided=p,
    )


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(order))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def merge_seed(root, seed, sessions):
    destination = root / f"evaluation-s{seed}"
    destination.mkdir(exist_ok=True)
    all_metrics, all_episodes, all_traces = [], [], []
    for folder, rename in (
        (f"variants-s{seed}", None),
        (f"continued-eval-s{seed}", "rmappo_continued"),
        (f"randomized-eval-s{seed}", "rmappo_randomized"),
    ):
        source = root / folder
        if not (source / "COMPLETE").exists():
            raise RuntimeError(f"Incomplete evaluation: {source}")
        metrics = json.loads((source / "session-metrics.json").read_text())
        with (source / "episodes.csv").open() as f:
            episodes = list(csv.DictReader(f))
        traces = [
            json.loads(line) for line in (source / "diagnostics.jsonl").read_text().splitlines()
        ]
        expected = {"frozen"} if rename else V2_MODES | {"frozen"}
        keys = {(r["mode"], r["condition"], int(r["session"])) for r in metrics}
        target = {
            (m, c, i)
            for m in expected
            for c in ("none", "semantic", "behavioral", "both")
            for i in range(sessions)
        }
        if keys != target or len(metrics) != len(target):
            raise RuntimeError(f"Missing or duplicate paired sessions: {source}")
        episode_keys = {
            (r["mode"], r["condition"], int(r["session"]), int(r["episode"])) for r in episodes
        }
        if (
            episode_keys != {(*key, e) for key in target for e in range(60)}
            or len(episodes) != len(target) * 60
        ):
            raise RuntimeError(f"Missing or duplicate episodes: {source}")
        if len(traces) != len(target) or any(len(t["diagnostics"]) != 60 for t in traces):
            raise RuntimeError(f"Incomplete diagnostics: {source}")
        for collection in (metrics, episodes, traces):
            for row in collection:
                if rename:
                    row["mode"] = rename
        all_metrics.extend(metrics)
        all_episodes.extend(episodes)
        all_traces.extend(traces)
    # Confirm paired seeds and change points across every method.
    for condition in ("none", "semantic", "behavioral", "both"):
        for i in range(sessions):
            paired = {
                (m["seed"], m["change_episode"])
                for m in all_metrics
                if m["condition"] == condition and m["session"] == i
            }
            if len(paired) != 1:
                raise RuntimeError("Unpaired evaluation seeds/change points")
    (destination / "session-metrics.json").write_text(json.dumps(all_metrics, indent=2))
    with (destination / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_episodes[0]))
        writer.writeheader()
        writer.writerows(all_episodes)
    (destination / "diagnostics.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in all_traces)
    )
    (destination / "COMPLETE").write_text("Validated complete merge\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sessions", type=int, default=20)
    args = parser.parse_args()
    for seed in range(1, args.seeds + 1):
        merge_seed(args.root, seed, args.sessions)
    spec = importlib.util.spec_from_file_location(
        "v2_summary", Path(__file__).with_name("summarize-adaptation-v2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.aggregate(args.root)
    data = json.loads((args.root / "training-seed-results.json").read_text())
    index = {(r["mode"], r["condition"], r["training_seed"]): r["post_return"] for r in data}
    modes = sorted({r["mode"] for r in data} - {"selective_v2"})
    comparisons = []
    for mode in modes:
        for condition in ("none", "semantic", "behavioral", "both", "shift_macro"):
            conditions = (
                ("semantic", "behavioral", "both") if condition == "shift_macro" else (condition,)
            )
            diffs = [
                float(
                    np.mean(
                        [index[("selective_v2", c, s)] - index[(mode, c, s)] for c in conditions]
                    )
                )
                for s in range(1, args.seeds + 1)
            ]
            comparisons.append(
                dict(
                    control=mode,
                    condition=condition,
                    seed_differences=diffs,
                    **paired_interval(diffs),
                )
            )
    primary_modes = {"motor_only", "joint_v2", "unguarded_v2"}
    primary = [
        r for r in comparisons if r["control"] in primary_modes and r["condition"] == "shift_macro"
    ]
    if args.seeds >= 2:
        for row, p in zip(primary, holm([r["p_one_sided"] for r in primary]), strict=True):
            row["holm_p"] = p
    safety = [r for r in comparisons if r["control"] in primary_modes and r["condition"] == "none"]
    success = args.seeds == 5 and all(r.get("holm_p", 1) < 0.05 and r["mean"] > 0 for r in primary)
    success = success and all(
        r["lower_one_sided95"] is not None and r["lower_one_sided95"] > -0.25 for r in safety
    )
    result = dict(primary_criterion_met=success, seed_count=args.seeds, comparisons=comparisons)
    (args.root / "paired-comparisons.json").write_text(json.dumps(result, indent=2))
    base = (args.root / "REPORT.md").read_text(encoding="utf-8")
    # Remove the older fixed three-seed interpretation; retain its raw tables.
    base = base.split("## Interpretation limits")[0]
    base = base.replace("# V2: exploratory multi-seed comparison", "# V2 versus simpler controls")
    base = base.replace("alarms ", "activation flags ")
    lines = [
        base,
        "## Preregistered paired analysis",
        "",
        f"Primary criterion met: **{success}**. See PROTOCOL.md for the exact criterion.",
        "",
        "Positive differences favor V2. Intervals use paired training-seed means.",
        "",
        "| Control | Endpoint | V2 minus control | 95% CI | Holm p (primary only) |",
        "|---|---|---:|---|---:|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['control']} | {r['condition']} | {r['mean']:.4f} "
            f"| {r['ci95']} | {r.get('holm_p', 'secondary')} |"
        )
    lines += [
        "",
        "## Normal-session actual adapter application",
        "",
        "A frozen actor can raise unused diagnostic flags. Count applied corrections separately.",
        "",
    ]
    for seed in range(1, args.seeds + 1):
        traces = [
            json.loads(t)
            for t in (args.root / f"evaluation-s{seed}/diagnostics.jsonl").read_text().splitlines()
        ]
        for mode in sorted({t["mode"] for t in traces}):
            normal = [t for t in traces if t["mode"] == mode and t["condition"] == "none"]
            applied = sum(any(d["adapter_applied"] for d in t["diagnostics"]) for t in normal)
            lines.append(
                f"- seed {seed}, {mode}: correction applied in "
                f"{applied}/{len(normal)} normal sessions."
            )
    lines += [
        "",
        "## Limits",
        "",
        "Five training seeds remain a small sample. Secondary comparisons are descriptive. "
        "No specialized published adaptation baseline has been reproduced here. "
        "R-MAPPO continuation/randomization use the official optimizer with our training wrapper; "
        "they receive rewards rather than V2's extra clean-symbol/change-label supervision. "
        "Matched V2 ablations share weights and supervision. The behavioral intervention remains "
        "an actuator rotation, not a general teammate replacement. Preserve negative findings.",
        "",
    ]
    (args.root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
