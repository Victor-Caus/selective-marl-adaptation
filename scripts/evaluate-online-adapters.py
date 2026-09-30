"""Paired evaluation in native MPE2, including label-free online adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config

from selective_marl.adapters import OnlineAdapters
from selective_marl.adapters_v2 import V2_MODES, GuardedAdapters, SemanticMemory
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.environments.reference_backend import HISTORICAL_TO_MPE2, ReferenceBackendEnv
from selective_marl.evaluation.benchmark import load_policy
from selective_marl.evaluation.metrics import compute_shift_metrics


def actor_from(path):
    args = get_config().parse_args([])
    args.algorithm_name = "rmappo"
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    probe = ReferenceBackendEnv()
    policy = R_MAPPOPolicy(
        args, probe.observation_space[0], probe.share_observation_space[0], probe.action_space[0]
    )
    probe.close()
    policy.actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    policy.actor.eval()
    return policy.actor, args


def session(
    actor,
    args,
    mode,
    condition,
    seed,
    episodes,
    change_episode,
    deterministic=False,
    memory=None,
    detector_threshold=6.0,
):
    torch.manual_seed(seed)
    is_v2 = mode in V2_MODES
    adapter = (
        GuardedAdapters(actor, memory=memory, mode=mode, threshold=detector_threshold)
        if is_v2
        else OnlineAdapters(actor, mode=mode)
    )
    env = ReferenceSessionEnv(make_session_spec(InterventionKind(condition), change_episode, seed))
    returns, diagnostics = [], []
    try:
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode, episode_index=episode)
            if is_v2:
                adapter.begin_episode()
            hidden = torch.zeros(2, args.recurrent_N, args.hidden_size)
            initial_hidden = hidden.clone()
            totals = np.zeros(2)
            applied = False
            trajectory = ([], [], [], [])
            while env.agents:
                obs = np.stack([observations[a] for a in env.possible_agents])
                with torch.no_grad():
                    move, message, hidden = adapter.distributions(obs, hidden)
                    applied = applied or bool(
                        not torch.allclose(adapter.motor, torch.eye(5).repeat(2, 1, 1))
                        or (
                            is_v2
                            and adapter.memory is not None
                            and np.any(adapter.semantic_active & (obs[:, -10:].sum(-1) > 0))
                        )
                    )
                    movement = move.probs.argmax(-1) if deterministic else move.sample()
                    symbol = message.probs.argmax(-1) if deterministic else message.sample()
                    actions = torch.stack([movement, symbol], dim=-1)
                    logs = move.log_prob(movement) + message.log_prob(symbol)
                indices = actions.numpy()
                product = HISTORICAL_TO_MPE2[indices[:, 0]] + 5 * indices[:, 1]
                observations, rewards, _, _, _ = env.step(
                    dict(zip(env.possible_agents, product.tolist(), strict=True))
                )
                next_obs = np.stack([observations[a] for a in env.possible_agents])
                reward = np.asarray([rewards[a] for a in env.possible_agents])
                if is_v2:
                    adapter.feedback(obs, indices, reward, next_obs)
                else:
                    adapter.observe_transition(obs, indices, next_obs)
                for buffer, value in zip(
                    trajectory, (obs, indices, logs.numpy(), reward), strict=True
                ):
                    buffer.append(value)
                totals += reward
            adapter.observe_return(totals)
            # All evaluated modes use identical stochastic action sampling. A separate
            # deterministic frozen run is a descriptive control, never a PPO rollout.
            if not deterministic:
                adapter.update(trajectory, initial_hidden)
            returns.append(float(totals.mean()))
            diagnostic = dict(
                semantic=adapter.semantic_active.tolist(),
                motor=adapter.motor_active.tolist(),
                adapter_applied=applied,
            )
            if is_v2:
                diagnostic.update(
                    rollbacks=adapter.rollbacks.tolist(), cusum=adapter.detector.cusum.tolist()
                )
            diagnostics.append(diagnostic)
    finally:
        env.close()
    return returns, diagnostics


def legacy_session(path, condition, seed, episodes, change_episode, deterministic):
    torch.manual_seed(seed)
    policy = load_policy(path, torch.device("cpu"))
    env = ReferenceSessionEnv(make_session_spec(InterventionKind(condition), change_episode, seed))
    returns = []
    try:
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode, episode_index=episode)
            hidden = policy.model.initial_hidden(2, torch.device("cpu"))
            total = 0.0
            while env.agents:
                obs = np.stack([observations[a] for a in env.possible_agents])
                state = np.repeat(env.global_state(observations)[None], 2, axis=0)
                with torch.no_grad():
                    output = policy.model.forward_step(
                        torch.tensor(obs), torch.tensor(state), hidden, deterministic=deterministic
                    )
                hidden = output.hidden
                observations, rewards, _, _, _ = env.step(
                    dict(zip(env.possible_agents, output.actions.tolist(), strict=True))
                )
                total += float(np.mean(list(rewards.values())))
            returns.append(total)
    finally:
        env.close()
    return returns, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--change-range", type=int, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--change-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--modes", nargs="+", default=["frozen", "joint", "selective"])
    parser.add_argument(
        "--conditions", nargs="+", default=["none", "semantic", "behavioral", "both"]
    )
    parser.add_argument("--deterministic", action="store_true")
    cli = parser.parse_args()
    if (cli.actor is None) == (cli.legacy_checkpoint is None):
        raise ValueError("Provide exactly one actor or legacy checkpoint")
    if cli.legacy_checkpoint and cli.modes != ["frozen"]:
        raise ValueError("Legacy comparison supports only frozen mode")
    if cli.deterministic and cli.modes != ["frozen"]:
        raise ValueError("Deterministic evaluation is supported only for the frozen control")
    if not set(cli.modes) <= {"frozen", "joint", "selective"} | V2_MODES:
        raise ValueError("Unknown adaptation mode")
    if any(m in V2_MODES - {"motor_only"} for m in cli.modes) and cli.memory is None:
        raise ValueError("Learned adapter modes require --memory")
    torch.set_num_threads(1)
    actor, args = actor_from(cli.actor) if cli.actor else (None, None)
    memory, threshold = None, 6.0
    if cli.memory:
        checkpoint = torch.load(cli.memory, map_location="cpu", weights_only=False)
        memory = SemanticMemory(checkpoint["hidden_size"])
        memory.load_state_dict(checkpoint["model_state"])
        memory.eval()
        threshold = checkpoint["threshold"]
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "config.json").write_text(
        json.dumps(
            {
                **vars(cli),
                "actor": str(cli.actor),
                "output": str(cli.output),
                "legacy_checkpoint": str(cli.legacy_checkpoint),
                "memory": str(cli.memory),
                "detector_threshold": threshold,
                "information": "each adapter uses its own local observation, action and reward",
                "state": "base GRU resets per episode; adapter state persists across episodes",
                "reward": "native MPE2 mean agent reward; identical for all modes",
            },
            indent=2,
        )
    )
    records, summaries = [], []
    with (cli.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mode", "condition", "session", "seed", "episode", "return"]
        )
        writer.writeheader()
        for mode in cli.modes:
            for condition in cli.conditions:
                for i in range(cli.sessions):
                    seed = cli.seed + i * 1000
                    change = (
                        int(
                            np.random.default_rng(seed).integers(
                                cli.change_range[0], cli.change_range[1] + 1
                            )
                        )
                        if cli.change_range
                        else cli.change_episode
                    )
                    if cli.legacy_checkpoint:
                        values, diagnostics = legacy_session(
                            cli.legacy_checkpoint,
                            condition,
                            seed,
                            cli.episodes,
                            change,
                            cli.deterministic,
                        )
                    else:
                        values, diagnostics = session(
                            actor,
                            args,
                            mode,
                            condition,
                            seed,
                            cli.episodes,
                            change,
                            cli.deterministic,
                            memory,
                            threshold,
                        )
                    for episode, value in enumerate(values):
                        writer.writerow(
                            dict(
                                mode=mode,
                                condition=condition,
                                session=i,
                                seed=seed,
                                episode=episode,
                                **{"return": value},
                            )
                        )
                    f.flush()
                    metrics = compute_shift_metrics(values, change_episode=change).to_dict()
                    row = dict(
                        mode=mode,
                        condition=condition,
                        session=i,
                        seed=seed,
                        change_episode=change,
                        **metrics,
                    )
                    records.append(row)
                    with (cli.output / "diagnostics.jsonl").open("a") as df:
                        df.write(
                            json.dumps(
                                dict(
                                    mode=mode,
                                    condition=condition,
                                    session=i,
                                    diagnostics=diagnostics,
                                )
                            )
                            + "\n"
                        )
                    print(json.dumps(row), flush=True)
                rows = [r for r in records if r["mode"] == mode and r["condition"] == condition]
                summary = dict(mode=mode, condition=condition, sessions=len(rows))
                for key in (
                    "pre_change_return",
                    "immediate_post_change_return",
                    "cumulative_regret",
                ):
                    data = np.asarray([r[key] for r in rows])
                    summary[key] = float(data.mean())
                    summary[key + "_ci95_halfwidth"] = (
                        float(1.96 * data.std(ddof=1) / np.sqrt(len(data)))
                        if len(data) > 1
                        else None
                    )
                summaries.append(summary)
                (cli.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    (cli.output / "session-metrics.json").write_text(json.dumps(records, indent=2))
    (cli.output / "COMPLETE").write_text("Evaluation finished\n")


if __name__ == "__main__":
    main()
