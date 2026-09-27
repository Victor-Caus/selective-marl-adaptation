"""Deterministically evaluate an official R-MAPPO checkpoint on pinned MPE."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import imageio
import numpy as np
import torch
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv


def build_args() -> argparse.Namespace:
    args = get_config().parse_args([])
    args.algorithm_name = "rmappo"
    args.scenario_name = "simple_reference"
    args.num_agents = 2
    args.num_landmarks = 3
    args.episode_length = 25
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    return args


def one_hot_actions(indices: np.ndarray, action_space: object, silence: bool) -> np.ndarray:
    branches = []
    for branch in range(action_space.shape):
        width = int(action_space.high[branch]) + 1
        encoded = np.eye(width, dtype=np.float32)[indices[:, branch]]
        if silence and branch == 1:
            encoded.fill(0.0)
        branches.append(encoded)
    return np.concatenate(branches, axis=1)


def run_condition(
    *,
    args: argparse.Namespace,
    actor_path: Path,
    condition: str,
    episodes: int,
    seed: int,
    gif_path=None,
) -> tuple[list[float], Counter[int]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = MPEEnv(args)
    env.seed(seed)
    device = torch.device("cpu")
    policy = R_MAPPOPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.actor.eval()

    returns: list[float] = []
    message_counts: Counter[int] = Counter()
    frames: list[np.ndarray] = []
    for episode in range(episodes):
        obs = env.reset()
        if gif_path is not None and episode == 0:
            frames.append(env.render("rgb_array")[0])
        rnn_states = np.zeros(
            (args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
        )
        masks = np.ones((args.num_agents, 1), dtype=np.float32)
        episode_return = 0.0
        for _step in range(args.episode_length):
            if condition == "random":
                action_indices = np.column_stack(
                    [
                        rng.integers(0, int(env.action_space[0].high[branch]) + 1, args.num_agents)
                        for branch in range(env.action_space[0].shape)
                    ]
                )
            else:
                with torch.no_grad():
                    actions, next_rnn_states = policy.act(
                        np.asarray(obs), rnn_states, masks, deterministic=True
                    )
                action_indices = actions.cpu().numpy().astype(int)
                rnn_states = next_rnn_states.cpu().numpy()
            message_counts.update(int(value) for value in action_indices[:, 1])
            env_actions = one_hot_actions(
                action_indices, env.action_space[0], silence=condition == "silenced"
            )
            obs, rewards, _dones, _infos = env.step(env_actions)
            episode_return += float(np.mean(rewards))
            if gif_path is not None and episode == 0:
                frames.append(env.render("rgb_array")[0])
        returns.append(episode_return)
    env.close()
    if gif_path is not None:
        imageio.mimsave(gif_path, frames, duration=0.1)
    return returns, message_counts


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "episodes": len(values),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "standard_error": standard_error,
        "ci95_half_width": 1.96 * standard_error,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def normalized_entropy(counts: Counter[int], alphabet_size: int = 10) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(alphabet_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--save-gif", action="store_true")
    cli = parser.parse_args()
    actor_path = cli.model_dir.resolve() / "actor.pt"
    if not actor_path.is_file():
        raise FileNotFoundError(actor_path)
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_args()

    raw: dict[str, list[float]] = {}
    messages: dict[str, Counter[int]] = {}
    for condition in ("trained", "silenced", "random"):
        raw[condition], messages[condition] = run_condition(
            args=args,
            actor_path=actor_path,
            condition=condition,
            episodes=cli.episodes,
            seed=cli.seed,
            gif_path=(output_dir / "trained-policy.gif")
            if cli.save_gif and condition == "trained"
            else None,
        )

    with (output_dir / "evaluation-episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "episode", "return"])
        for condition, values in raw.items():
            writer.writerows((condition, index, value) for index, value in enumerate(values))

    summary = {condition: describe(values) for condition, values in raw.items()}
    summary["trained"]["message_entropy_normalized"] = normalized_entropy(messages["trained"])
    summary["trained"]["message_counts"] = dict(sorted(messages["trained"].items()))
    communication_differences = [
        raw["trained"][index] - raw["silenced"][index]
        for index in range(len(raw["trained"]))
    ]
    random_differences = [
        raw["trained"][index] - raw["random"][index]
        for index in range(len(raw["trained"]))
    ]
    summary["communication_ablation"] = describe(communication_differences)
    summary["trained_vs_random"] = describe(random_differences)
    (output_dir / "evaluation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Deterministic checkpoint evaluation", ""]
    for condition in ("trained", "silenced", "random"):
        result = summary[condition]
        lines.append(
            f"- {condition}: {result['mean']:.6f} +/- {result['ci95_half_width']:.6f} "
            f"(95% CI half-width, n={result['episodes']})"
        )
    difference = summary["communication_ablation"]
    random_difference = summary["trained_vs_random"]
    lines.extend(
        [
            f"- trained minus silenced: {difference['mean']:.6f} +/- "
            f"{difference['ci95_half_width']:.6f} (paired 95% CI half-width)",
            f"- trained minus random: {random_difference['mean']:.6f} +/- "
            f"{random_difference['ci95_half_width']:.6f} (paired 95% CI half-width)",
            f"- normalized message entropy: {summary['trained']['message_entropy_normalized']:.6f}",
            "",
            "The silenced condition keeps the trained physical action but replaces the emitted",
            "communication vector with zeros. Random is an untrained factorized-action policy.",
        ]
    )
    (output_dir / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
