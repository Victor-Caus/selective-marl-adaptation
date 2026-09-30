"""Validate the new single-controlled-agent experiment before comparing results."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--steps", type=int, default=40000000)
    cli = p.parse_args()
    modes = [
        "frozen",
        "motor_only",
        "joint_v2",
        "unguarded_v2",
        "selective_v2",
        "liam_matched",
        "liam_long",
    ]
    conditions = ["none", "semantic", "behavioral", "both"]
    seed_rows, records = [], []
    for seed in range(1, cli.seeds + 1):
        train = json.loads((cli.root / f"train-s{seed}/status.json").read_text())
        assert train["status"] == "complete" and train["completed_steps"] == cli.steps
        folder = cli.root / f"evaluation-s{seed}"
        assert (folder / "COMPLETE").exists()
        rows = json.loads((folder / "session-metrics.json").read_text())
        expected = {(m, c, s) for m in modes for c in conditions for s in range(cli.sessions)}
        assert len(rows) == len(expected)
        assert {(r["mode"], r["condition"], r["session"]) for r in rows} == expected
        with (folder / "episodes.csv").open() as f:
            episodes = list(csv.DictReader(f))
        assert len(episodes) == len(expected) * 60
        assert {
            (r["mode"], r["condition"], int(r["session"]), int(r["episode"])) for r in episodes
        } == {(*key, e) for key in expected for e in range(60)}
        assert all(np.isfinite(float(r["return"])) for r in episodes)
        for c in conditions:
            for s in range(cli.sessions):
                assert (
                    len(
                        {
                            (r["seed"], r["change_episode"])
                            for r in rows
                            if r["condition"] == c and r["session"] == s
                        }
                    )
                    == 1
                )
        for mode in modes:
            for condition in conditions:
                selected = [r for r in rows if r["mode"] == mode and r["condition"] == condition]
                seed_rows.append(
                    dict(
                        training_seed=seed,
                        mode=mode,
                        condition=condition,
                        pre_return=float(np.mean([r["pre_return"] for r in selected])),
                        post_return=float(np.mean([r["post_return"] for r in selected])),
                        sessions_with_adapter=sum(r["applied_episodes"] > 0 for r in selected),
                    )
                )
        records.extend(dict(training_seed=seed, **r) for r in rows)
    (cli.root / "training-seed-results.json").write_text(json.dumps(seed_rows, indent=2))
    (cli.root / "session-results.json").write_text(json.dumps(records, indent=2))
    spec = importlib.util.spec_from_file_location(
        "statistics_helpers", Path(__file__).with_name("summarize-selectivity-controls.py")
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    index = {(r["mode"], r["condition"], r["training_seed"]): r for r in seed_rows}
    comparisons = []
    for control in ("liam_matched", "liam_long"):
        for condition in conditions:
            differences = [
                index[("selective_v2", condition, s)]["post_return"]
                - index[(control, condition, s)]["post_return"]
                for s in range(1, cli.seeds + 1)
            ]
            comparisons.append(
                dict(
                    control=control,
                    condition=condition,
                    seed_differences=differences,
                    **helpers.paired_interval(differences),
                )
            )
    primary = [r for r in comparisons if r["control"] == "liam_long" and r["condition"] != "none"]
    if cli.seeds > 1:
        for row, value in zip(
            primary, helpers.holm([r["p_one_sided"] for r in primary]), strict=True
        ):
            row["holm_p"] = value
    safety = next(
        r for r in comparisons if r["control"] == "liam_long" and r["condition"] == "none"
    )
    passed = cli.seeds == 5 and all(r.get("holm_p", 1) < 0.05 and r["mean"] > 0 for r in primary)
    passed = (
        passed and safety["lower_one_sided95"] is not None and safety["lower_one_sided95"] > -0.25
    )
    (cli.root / "paired-comparisons.json").write_text(
        json.dumps(dict(primary_criterion_met=passed, comparisons=comparisons), indent=2)
    )
    lines = [
        "# Official LIAM core port versus V2: fixed-partner evaluation",
        "",
        "One controlled agent for every method. "
        "These scores must not be mixed with the old two-adapter tables.",
        "",
        f"Primary criterion met: **{passed}**. Training seeds: {cli.seeds}.",
        "",
        "| Method | Condition | Pre-return | Post-return mean ± SD | Post minus pre |",
        "|---|---|---:|---:|---:|",
    ]
    for mode in modes:
        for condition in conditions:
            values = [r for r in seed_rows if r["mode"] == mode and r["condition"] == condition]
            pre = np.mean([r["pre_return"] for r in values])
            post = np.array([r["post_return"] for r in values])
            sd = post.std(ddof=1) if len(post) > 1 else 0
            lines.append(
                f"| {mode} | {condition} | {pre:.3f} | {post.mean():.3f} ± {sd:.3f} "
                f"| {post.mean() - pre:.3f} |"
            )
    lines += [
        "",
        "## Paired V2-minus-LIAM post-return",
        "",
        "| Control | Condition | Difference | 95% CI | Holm p (primary only) |",
        "|---|---|---:|---|---:|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['control']} | {r['condition']} | {r['mean']:.4f} "
            f"| {r['ci95']} | {r.get('holm_p', 'secondary')} |"
        )
    normal = [r for r in records if r["mode"] == "selective_v2" and r["condition"] == "none"]
    lines += [
        "",
        "V2 normal sessions with applied corrections: "
        f"{sum(r['applied_episodes'] > 0 for r in normal)}/{len(normal)}.",
        "",
        "## Limits",
        "",
        "This imports the official LIAM optimizer and networks with dimension/environment changes. "
        "It does not reproduce the original environment, partner population, or published scores. "
        "LIAM uses teammate reconstruction targets during training; V2 uses clean-symbol labels. "
        "LIAM recurrence resets each episode; V2's semantic context persists. "
        "The long checkpoint gets 40M learner interactions; the earlier checkpoint gets 3.4208M, "
        "with shared partner-pretraining costs additional to LIAM. "
        "Five seeds and one actuator-rotation task do not establish general SOTA superiority. "
        "Consider pre-shift competence alongside post-shift scores. See PROTOCOL.md.",
        "",
    ]
    (cli.root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
