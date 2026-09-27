"""Convert an official MAPPO TensorBoardX export into reviewable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def metric_name(raw_name: str) -> str:
    normalized = raw_name.replace("\\", "/").rstrip("/")
    parts = normalized.split("/")
    if "logs" in parts:
        parts = parts[parts.index("logs") + 1 :]
    midpoint = len(parts) // 2
    if len(parts) % 2 == 0 and parts[:midpoint] == parts[midpoint:]:
        parts = parts[:midpoint]
    return "/".join(parts)


def load_series(run_dir: Path) -> dict[str, list[list[float]]]:
    summaries = sorted(run_dir.rglob("summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json found below {run_dir}")
    raw = json.loads(summaries[-1].read_text(encoding="utf-8"))
    series: dict[str, list[list[float]]] = {}
    for raw_name, values in raw.items():
        name = metric_name(raw_name)
        if name in series:
            raise ValueError(f"Duplicate normalized metric name: {name}")
        series[name] = values
    return series


def summarize(values: list[list[float]]) -> dict[str, float | int]:
    ordered = sorted(values, key=lambda item: item[1])
    numeric = [float(item[2]) for item in ordered]
    window = max(1, len(numeric) // 5)
    early_mean = sum(numeric[:window]) / window
    late_mean = sum(numeric[-window:]) / window
    return {
        "points": len(ordered),
        "first_step": int(ordered[0][1]),
        "last_step": int(ordered[-1][1]),
        "first": numeric[0],
        "last": numeric[-1],
        "minimum": min(numeric),
        "maximum": max(numeric),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_minus_early": late_mean - early_mean,
    }


def write_plot(path: Path, reward_values: list[list[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ordered = sorted(reward_values, key=lambda item: item[1])
    steps = [int(item[1]) for item in ordered]
    rewards = [float(item[2]) for item in ordered]
    plt.figure(figsize=(9, 5))
    plt.plot(steps, rewards, linewidth=1.4, label="logged mean episode return")
    plt.xlabel("environment steps")
    plt.ylabel("average episode reward")
    plt.title("Official R-MAPPO - MPE simple_reference")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    series = load_series(run_dir)

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "wall_time", "step", "value"])
        for name, values in sorted(series.items()):
            for wall_time, step, value in sorted(values, key=lambda item: item[1]):
                writer.writerow([name, wall_time, int(step), value])

    analysis = {name: summarize(values) for name, values in sorted(series.items()) if values}
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reward = analysis.get("average_episode_rewards")
    lines = ["# Reference run analysis", ""]
    if reward:
        lines.extend(
            [
                f"- Logged steps: {reward['first_step']} to {reward['last_step']}",
                f"- First logged reward: {reward['first']:.6f}",
                f"- Last logged reward: {reward['last']:.6f}",
                f"- Early-window mean: {reward['early_mean']:.6f}",
                f"- Late-window mean: {reward['late_mean']:.6f}",
                f"- Late minus early: {reward['late_minus_early']:.6f}",
                f"- Best logged reward: {reward['maximum']:.6f}",
                "",
                "A less-negative reward is better in this environment. This training curve",
                "alone does not replace deterministic checkpoint evaluation or a multi-seed",
                "confidence interval.",
            ]
        )
        write_plot(run_dir / "learning-curve.png", series["average_episode_rewards"])
    else:
        lines.append("The TensorBoard export did not contain average_episode_rewards.")
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
