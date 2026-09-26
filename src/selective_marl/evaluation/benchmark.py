"""Evaluation, plotting, video generation, and diagnostic export."""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from selective_marl.artifacts import RunArtifacts
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.evaluation.metrics import compute_shift_metrics
from selective_marl.models import MAPPOPolicy, build_policy
from selective_marl.training import choose_device, set_seed


@dataclass(slots=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505)
    sessions_per_condition: int = 5
    session_episodes: int = 20
    change_episode: int = 10
    max_cycles: int = 25
    deterministic: bool = True
    device: str = "auto"
    record_video: bool = True
    video_condition: str = "semantic"


@dataclass(slots=True)
class LoadedPolicy:
    name: str
    model: MAPPOPolicy | None
    mask_messages: bool = False
    oracle_gate: bool = False


def load_policy(path: Path, device: torch.device, *, oracle_gate: bool = False) -> LoadedPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    algorithm = checkpoint["algorithm"]
    hidden_size = int(checkpoint["config"].get("hidden_size", 128))
    if algorithm == "maddpg":
        from selective_marl.maddpg import MADDPGPolicy

        model = MADDPGPolicy(hidden_size).to(device)
    else:
        model = build_policy(algorithm, hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    name = f"{algorithm}-oracle" if oracle_gate else algorithm
    return LoadedPolicy(name, model, algorithm == "mappo_no_comm", oracle_gate)


def _random_actions(env: ReferenceSessionEnv, rng: np.random.Generator) -> dict[str, int]:
    return {agent: int(rng.integers(env.action_space(agent).n)) for agent in env.agents}


def evaluate(
    checkpoints: list[Path],
    config: EvaluationConfig,
    output_root: Path,
    *,
    include_random: bool = True,
) -> Path:
    device = choose_device(config.device)
    artifacts = RunArtifacts(output_root, "evaluation")
    artifacts.metadata()
    artifacts.write_json("config.json", asdict(config))
    policies: list[LoadedPolicy] = []
    if include_random:
        policies.append(LoadedPolicy("random", None))
    for checkpoint in checkpoints:
        loaded = load_policy(checkpoint, device)
        policies.append(loaded)
        if loaded.model is not None and loaded.model.selective:
            policies.append(load_policy(checkpoint, device, oracle_gate=True))

    for index, checkpoint in enumerate(checkpoints):
        source = checkpoint.parent
        destination = artifacts.path / "training" / f"{index:02d}-{source.name}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "config.json",
            "metadata.json",
            "train_summary.json",
            "train_metrics.jsonl",
            "episodes.jsonl",
        ):
            source_file = source / name
            if source_file.exists():
                shutil.copy2(source_file, destination / name)

    episode_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    diagnosis_rows: list[dict[str, Any]] = []
    videos_written: set[str] = set()
    for policy in policies:
        for condition in InterventionKind:
            for session_number in range(config.sessions_per_condition):
                seed = config.seeds[session_number % len(config.seeds)] + 10_000 * session_number
                set_seed(seed)
                rng = np.random.default_rng(seed)
                spec = make_session_spec(condition, config.change_episode, seed)
                should_video = (
                    config.record_video
                    and condition.value == config.video_condition
                    and policy.name not in videos_written
                )
                env = ReferenceSessionEnv(
                    spec,
                    max_cycles=config.max_cycles,
                    render_mode="rgb_array" if should_video else None,
                    mask_messages=policy.mask_messages,
                )
                hidden = (
                    policy.model.initial_hidden(2, device) if policy.model is not None else None
                )
                session_returns: list[float] = []
                frames: list[Image.Image] = []
                true_labels: list[int] = []
                predictions: list[int] = []
                try:
                    for episode in range(config.session_episodes):
                        observations, _ = env.reset(seed=seed + episode, episode_index=episode)
                        totals = {agent: 0.0 for agent in env.possible_agents}
                        if should_video and episode in {
                            max(0, config.change_episode - 1),
                            config.change_episode,
                        }:
                            frame = env.render()
                            if frame is not None:
                                frames.append(Image.fromarray(frame))
                        while env.agents:
                            if policy.model is None:
                                actions = _random_actions(env, rng)
                                predicted = 0
                            else:
                                agents = env.possible_agents
                                obs = np.stack([observations[a] for a in agents])
                                state = np.repeat(
                                    env.global_state(observations)[None, :], len(agents), axis=0
                                )
                                labels = torch.full(
                                    (len(agents),),
                                    env.true_label(),
                                    dtype=torch.long,
                                    device=device,
                                )
                                with torch.no_grad():
                                    output = policy.model.forward_step(
                                        torch.as_tensor(obs, dtype=torch.float32, device=device),
                                        torch.as_tensor(state, dtype=torch.float32, device=device),
                                        hidden,
                                        labels,
                                        deterministic=config.deterministic,
                                        oracle_gate=policy.oracle_gate,
                                    )
                                hidden = output.hidden
                                actions = {
                                    agent: int(output.actions[index])
                                    for index, agent in enumerate(agents)
                                }
                                predicted = (
                                    int(output.diagnosis_logits.argmax(-1)[0])
                                    if output.diagnosis_logits is not None
                                    else 0
                                )
                            observations, rewards, terminations, truncations, _ = env.step(actions)
                            for agent, reward in rewards.items():
                                totals[agent] += float(reward)
                            true_labels.append(env.true_label())
                            predictions.append(predicted)
                            if should_video and episode in {
                                max(0, config.change_episode - 1),
                                config.change_episode,
                            }:
                                frame = env.render()
                                if frame is not None:
                                    frames.append(Image.fromarray(frame))
                        team_return = float(np.mean(list(totals.values())))
                        session_returns.append(team_return)
                        episode_rows.append(
                            {
                                "algorithm": policy.name,
                                "condition": condition.value,
                                "session": session_number,
                                "seed": seed,
                                "episode": episode,
                                "phase": "post" if episode >= config.change_episode else "pre",
                                "team_return": team_return,
                            }
                        )
                    metrics = compute_shift_metrics(
                        session_returns,
                        change_episode=config.change_episode,
                        pre_window=min(5, config.change_episode),
                        post_window=min(5, config.session_episodes - config.change_episode),
                        recovery_sustain=min(3, config.session_episodes - config.change_episode),
                    )
                    pre_change_steps = config.change_episode * config.max_cycles
                    scored_true_labels = (
                        true_labels
                        if condition == InterventionKind.NONE
                        else true_labels[pre_change_steps:]
                    )
                    scored_predictions = (
                        predictions
                        if condition == InterventionKind.NONE
                        else predictions[pre_change_steps:]
                    )
                    diagnosis_accuracy = float(
                        np.mean(np.asarray(scored_true_labels) == np.asarray(scored_predictions))
                    )
                    false_alarm_predictions = (
                        predictions
                        if condition == InterventionKind.NONE
                        else predictions[:pre_change_steps]
                    )
                    false_alarm_rate = float(np.mean(np.asarray(false_alarm_predictions) != 0))
                    session_rows.append(
                        {
                            "algorithm": policy.name,
                            "condition": condition.value,
                            "session": session_number,
                            "seed": seed,
                            "diagnosis_accuracy": diagnosis_accuracy,
                            "false_alarm_rate": false_alarm_rate,
                            **metrics.to_dict(),
                        }
                    )
                    for expected in range(4):
                        for predicted in range(4):
                            count = sum(
                                int(a == expected and b == predicted)
                                for a, b in zip(true_labels, predictions, strict=True)
                            )
                            diagnosis_rows.append(
                                {
                                    "algorithm": policy.name,
                                    "condition": condition.value,
                                    "expected": expected,
                                    "predicted": predicted,
                                    "count": count,
                                }
                            )
                    if should_video and frames:
                        video_path = (
                            artifacts.path / "videos" / f"{policy.name}-{condition.value}.gif"
                        )
                        frames[0].save(
                            video_path,
                            save_all=True,
                            append_images=frames[1:],
                            duration=140,
                            loop=0,
                        )
                        videos_written.add(policy.name)
                finally:
                    env.close()

    artifacts.write_csv("episodes.csv", episode_rows)
    artifacts.write_csv("sessions.csv", session_rows)
    artifacts.write_csv("confusion_matrix.csv", diagnosis_rows)
    summary = _summarize(session_rows)
    artifacts.write_csv("summary.csv", summary)
    artifacts.write_json("summary.json", summary)
    artifacts.write_csv("comparisons-vs-mappo.csv", _comparisons(summary))
    _plot_returns(episode_rows, artifacts.path / "plots" / "return-curves.png")
    _plot_recovery(summary, artifacts.path / "plots" / "recovery-delay.png")
    _write_report(artifacts.path / "REPORT.md", summary, config)
    artifacts.bundle()
    return artifacts.path


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["condition"])].append(row)
    result: list[dict[str, Any]] = []
    for (algorithm, condition), group in sorted(grouped.items()):
        delays = [row["recovery_delay"] for row in group if row["recovery_delay"] is not None]
        pre = _mean_ci([r["pre_change_return"] for r in group])
        post = _mean_ci([r["immediate_post_change_return"] for r in group])
        drop = _mean_ci([r["performance_drop"] for r in group])
        regret = _mean_ci([r["cumulative_regret"] for r in group])
        result.append(
            {
                "algorithm": algorithm,
                "condition": condition,
                "runs": len(group),
                "pre_change_return_mean": pre[0],
                "pre_change_return_std": pre[1],
                "pre_change_return_ci95_low": pre[2],
                "pre_change_return_ci95_high": pre[3],
                "post_change_return_mean": post[0],
                "post_change_return_std": post[1],
                "post_change_return_ci95_low": post[2],
                "post_change_return_ci95_high": post[3],
                "performance_drop_mean": drop[0],
                "performance_drop_std": drop[1],
                "performance_drop_ci95_low": drop[2],
                "performance_drop_ci95_high": drop[3],
                "cumulative_regret_mean": regret[0],
                "cumulative_regret_std": regret[1],
                "cumulative_regret_ci95_low": regret[2],
                "cumulative_regret_ci95_high": regret[3],
                "recovery_delay_mean": float(np.mean(delays)) if delays else None,
                "recovery_rate": len(delays) / len(group),
                "diagnosis_accuracy_mean": float(np.mean([r["diagnosis_accuracy"] for r in group])),
                "false_alarm_rate_mean": float(np.mean([r["false_alarm_rate"] for r in group])),
            }
        )
    return result


def _mean_ci(values: list[float], samples: int = 2_000) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if len(array) == 1:
        return mean, std, mean, mean
    rng = np.random.default_rng(20260926)
    bootstrap = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return mean, std, float(low), float(high)


def _comparisons(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {row["condition"]: row for row in summary if row["algorithm"] == "mappo"}
    rows: list[dict[str, Any]] = []
    for row in summary:
        baseline = baselines.get(row["condition"])
        if baseline is None or row["algorithm"] == "mappo":
            continue
        rows.append(
            {
                "algorithm": row["algorithm"],
                "condition": row["condition"],
                "performance_drop_delta_vs_mappo": row["performance_drop_mean"]
                - baseline["performance_drop_mean"],
                "cumulative_regret_delta_vs_mappo": row["cumulative_regret_mean"]
                - baseline["cumulative_regret_mean"],
                "diagnosis_accuracy_delta_vs_mappo": row["diagnosis_accuracy_mean"]
                - baseline["diagnosis_accuracy_mean"],
            }
        )
    return rows


def _plot_returns(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["condition"], row["episode"])].append(row["team_return"])
    conditions = list(InterventionKind)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, conditions, strict=True):
        algorithms = sorted({key[0] for key in grouped if key[1] == condition.value})
        for algorithm in algorithms:
            episodes = sorted(key[2] for key in grouped if key[:2] == (algorithm, condition.value))
            means = [
                np.mean(grouped[(algorithm, condition.value, episode)]) for episode in episodes
            ]
            axis.plot(episodes, means, label=algorithm)
        axis.set_title(condition.value)
        axis.set_xlabel("episode")
        axis.set_ylabel("team return")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Simple Reference: performance around controlled shifts")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_recovery(summary: list[dict[str, Any]], path: Path) -> None:
    filtered = [row for row in summary if row["condition"] != "none"]
    labels = [f"{row['algorithm']}\n{row['condition']}" for row in filtered]
    values = [row["recovery_delay_mean"] or 0 for row in filtered]
    figure, axis = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5))
    axis.bar(range(len(labels)), values)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylabel("mean recovery delay (episodes)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_report(path: Path, summary: list[dict[str, Any]], config: EvaluationConfig) -> None:
    lines = [
        "# Evaluation report",
        "",
        "This report is generated from raw episode records. Short or single-seed runs are",
        "engineering checks and must not be interpreted as scientific evidence.",
        "",
        f"Sessions per condition: {config.sessions_per_condition}",
        f"Episodes per session: {config.session_episodes}",
        f"Change episode: {config.change_episode}",
        "",
        "## Aggregate results",
        "",
        "| Algorithm | Condition | Drop | Regret | Recovery | Diagnosis | False alarms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        delay = "n/a" if row["recovery_delay_mean"] is None else f"{row['recovery_delay_mean']:.2f}"
        lines.append(
            f"| {row['algorithm']} | {row['condition']} | "
            f"{row['performance_drop_mean']:.3f} | {row['cumulative_regret_mean']:.3f} | "
            f"{delay} ({row['recovery_rate']:.0%}) | "
            f"{row['diagnosis_accuracy_mean']:.2%} | "
            f"{row['false_alarm_rate_mean']:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
