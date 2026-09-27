"""Evaluate the pinned official R-MAPPO actor under controlled mechanism shifts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv

from selective_marl.reference_protocol import (
    permute_received_messages,
    rotate_teammate_movement,
)

CONDITIONS = ("none", "semantic", "behavioral", "both")
MESSAGE_COUNT = 10


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


def one_hot_actions(indices: np.ndarray, action_space: object) -> np.ndarray:
    branches = []
    for branch in range(action_space.shape):
        width = int(action_space.high[branch]) + 1
        branches.append(np.eye(width, dtype=np.float32)[indices[:, branch]])
    return np.concatenate(branches, axis=1)


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_half_width": 1.96 * standard_error,
    }


def session_metrics(returns: list[float], change_episode: int) -> dict[str, float | int | None]:
    values = np.asarray(returns, dtype=np.float64)
    pre = float(values[change_episode - 5 : change_episode].mean())
    post = float(values[change_episode : change_episode + 5].mean())
    regret = float(np.maximum(0.0, pre - values[change_episode:]).sum())
    target = pre * 0.95
    recovery = None
    for episode in range(change_episode, len(values) - 2):
        if bool(np.all(values[episode : episode + 3] >= target)):
            recovery = episode - change_episode
            break
    return {
        "pre_change_return": pre,
        "immediate_post_change_return": post,
        "performance_drop": pre - post,
        "cumulative_regret": regret,
        "recovery_delay": recovery,
    }


def run_session(
    policy: R_MAPPOPolicy,
    args: argparse.Namespace,
    condition: str,
    session_seed: int,
    episodes: int,
    change_episode: int,
) -> list[float]:
    torch.manual_seed(session_seed)
    np.random.seed(session_seed)
    rng = np.random.default_rng(session_seed)
    permutation = rng.permutation(MESSAGE_COUNT)
    if np.array_equal(permutation, np.arange(MESSAGE_COUNT)):
        permutation = np.roll(permutation, 1)
    env = MPEEnv(args)
    env.seed(session_seed)
    rnn_states = np.zeros(
        (args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    masks = np.ones((args.num_agents, 1), dtype=np.float32)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            active = episode >= change_episode
            obs = np.asarray(env.reset(), dtype=np.float32)
            if active and condition in {"semantic", "both"}:
                obs = permute_received_messages(obs, permutation)
            episode_return = 0.0
            for _step in range(args.episode_length):
                with torch.no_grad():
                    actions, next_rnn_states = policy.act(
                        obs, rnn_states, masks, deterministic=True
                    )
                action_indices = actions.cpu().numpy().astype(int)
                rnn_states = next_rnn_states.cpu().numpy()
                executed_actions = action_indices
                if active and condition in {"behavioral", "both"}:
                    executed_actions = rotate_teammate_movement(action_indices)
                obs, rewards, _dones, _infos = env.step(
                    one_hot_actions(executed_actions, env.action_space[0])
                )
                obs = np.asarray(obs, dtype=np.float32)
                if active and condition in {"semantic", "both"}:
                    obs = permute_received_messages(obs, permutation)
                episode_return += float(np.mean(rewards))
            returns.append(episode_return)
    finally:
        env.close()
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--change-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1001)
    cli = parser.parse_args()
    if not 5 <= cli.change_episode <= cli.episodes - 5:
        raise ValueError("change_episode must leave five episodes on each side")

    args = build_args()
    probe_env = MPEEnv(args)
    policy = R_MAPPOPolicy(
        args,
        probe_env.observation_space[0],
        probe_env.share_observation_space[0],
        probe_env.action_space[0],
        device=torch.device("cpu"),
    )
    probe_env.close()
    actor_path = cli.model_dir.resolve() / "actor.pt"
    policy.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    policy.actor.eval()

    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict] = []
    metric_rows: list[dict] = []
    for condition in CONDITIONS:
        for session in range(cli.sessions):
            session_seed = cli.seed + session
            returns = run_session(
                policy,
                args,
                condition,
                session_seed,
                cli.episodes,
                cli.change_episode,
            )
            episode_rows.extend(
                {
                    "condition": condition,
                    "session": session,
                    "seed": session_seed,
                    "episode": episode,
                    "phase": "post" if episode >= cli.change_episode else "pre",
                    "return": value,
                }
                for episode, value in enumerate(returns)
            )
            metric_rows.append(
                {
                    "condition": condition,
                    "session": session,
                    "seed": session_seed,
                    **session_metrics(returns, cli.change_episode),
                }
            )

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    with (output_dir / "session-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {}
    metric_names = (
        "pre_change_return",
        "immediate_post_change_return",
        "performance_drop",
        "cumulative_regret",
    )
    for condition in CONDITIONS:
        rows = [row for row in metric_rows if row["condition"] == condition]
        recovered = [row["recovery_delay"] for row in rows if row["recovery_delay"] is not None]
        summary[condition] = {
            name: describe([float(row[name]) for row in rows]) for name in metric_names
        }
        summary[condition]["recovery_rate"] = len(recovered) / len(rows)
        summary[condition]["recovery_delay"] = (
            describe([float(value) for value in recovered]) if recovered else None
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_repository": "https://github.com/marlbenchmark/on-policy",
                "source_commit": "de66d7a4b23fac2513f56f96f73b3f5cb96695ac",
                "checkpoint": str(actor_path),
                "sessions_per_condition": cli.sessions,
                "episodes_per_session": cli.episodes,
                "change_episode": cli.change_episode,
                "base_seed": cli.seed,
                "conditions": CONDITIONS,
                "recurrent_state_persists_across_episodes": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
