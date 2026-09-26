"""Compact MAPPO training loop for the controlled benchmark."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from selective_marl.artifacts import RunArtifacts
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.models import build_policy


@dataclass(slots=True)
class TrainConfig:
    algorithm: str = "mappo"
    seed: int = 42
    total_steps: int = 100_000
    rollout_steps: int = 1024
    update_epochs: int = 6
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    diagnosis_coef: float = 0.2
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    max_cycles: int = 25
    session_episodes: int = 20
    change_episode: int = 10
    log_interval: int = 1
    device: str = "auto"


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: TrainConfig, output_root: Path) -> Path:
    if config.algorithm == "maddpg":
        from selective_marl.maddpg import train_maddpg

        return train_maddpg(config, output_root)
    set_seed(config.seed)
    device = choose_device(config.device)
    artifacts = RunArtifacts(output_root, f"train-{config.algorithm}-seed{config.seed}")
    artifacts.metadata()
    artifacts.write_json("config.json", asdict(config))
    model = build_policy(config.algorithm, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    conditions = list(InterventionKind)
    session_index = 0
    episode_index = 0
    episode_seed = config.seed * 100_000
    initial_condition = (
        InterventionKind.SEMANTIC
        if config.algorithm == "channel_randomized"
        else InterventionKind.NONE
    )
    initial_change = 0 if config.algorithm == "channel_randomized" else config.change_episode
    spec = make_session_spec(initial_condition, initial_change, episode_seed)
    env = ReferenceSessionEnv(
        spec,
        max_cycles=config.max_cycles,
        mask_messages=config.algorithm == "mappo_no_comm",
    )
    observations, _ = env.reset(seed=episode_seed, episode_index=episode_index)
    agents = env.possible_agents
    hidden = model.initial_hidden(len(agents), device)
    episode_returns = {agent: 0.0 for agent in agents}
    completed_returns: list[float] = []
    global_step = 0
    update_index = 0
    started = time.perf_counter()

    try:
        while global_step < config.total_steps:
            buffer: dict[str, list] = {
                key: []
                for key in (
                    "obs",
                    "states",
                    "actions",
                    "log_probs",
                    "values",
                    "rewards",
                    "dones",
                    "labels",
                    "hidden_a",
                    "hidden_b",
                )
            }
            for _ in range(config.rollout_steps):
                obs_array = np.stack([observations[a] for a in agents])
                state = env.global_state(observations)
                state_array = np.repeat(state[None, :], len(agents), axis=0)
                labels = np.full(len(agents), env.true_label(), dtype=np.int64)
                obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
                state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=device)
                label_tensor = torch.as_tensor(labels, device=device)
                with torch.no_grad():
                    output = model.forward_step(
                        obs_tensor,
                        state_tensor,
                        hidden,
                        label_tensor,
                        oracle_gate=model.selective,
                    )
                buffer["obs"].append(obs_array)
                buffer["states"].append(state_array)
                buffer["actions"].append(output.actions.cpu().numpy())
                buffer["log_probs"].append(output.log_probs.cpu().numpy())
                buffer["values"].append(output.values.cpu().numpy())
                buffer["labels"].append(labels)
                if isinstance(hidden, tuple):
                    buffer["hidden_a"].append(hidden[0].detach().cpu().numpy())
                    buffer["hidden_b"].append(hidden[1].detach().cpu().numpy())
                elif hidden is not None:
                    buffer["hidden_a"].append(hidden.detach().cpu().numpy())
                hidden = output.hidden

                action_dict = {agent: int(output.actions[i]) for i, agent in enumerate(agents)}
                next_observations, rewards, terminations, truncations, _ = env.step(action_dict)
                dones = np.asarray(
                    [terminations.get(a, False) or truncations.get(a, False) for a in agents],
                    dtype=np.float32,
                )
                reward_array = np.asarray([rewards[a] for a in agents], dtype=np.float32)
                buffer["rewards"].append(reward_array)
                buffer["dones"].append(dones)
                for agent in agents:
                    episode_returns[agent] += float(rewards[agent])
                global_step += len(agents)
                observations = next_observations

                if bool(np.all(dones)):
                    team_return = float(np.mean(list(episode_returns.values())))
                    completed_returns.append(team_return)
                    artifacts.append_jsonl(
                        "episodes.jsonl",
                        {
                            "global_step": global_step,
                            "session": session_index,
                            "episode": episode_index,
                            "condition": spec.condition.value,
                            "label": spec.label(episode_index).value,
                            "team_return": team_return,
                        },
                    )
                    episode_index += 1
                    new_session = episode_index >= config.session_episodes
                    if new_session:
                        session_index += 1
                        episode_index = 0
                        if config.algorithm == "channel_randomized":
                            condition = InterventionKind.SEMANTIC
                            change_episode = 0
                        elif config.algorithm in {"rmappo", "selective"}:
                            condition = conditions[session_index % len(conditions)]
                            change_episode = config.change_episode
                        else:
                            condition = InterventionKind.NONE
                            change_episode = config.change_episode
                        env.close()
                        spec = make_session_spec(
                            condition, change_episode, config.seed + session_index
                        )
                        env = ReferenceSessionEnv(
                            spec,
                            max_cycles=config.max_cycles,
                            mask_messages=config.algorithm == "mappo_no_comm",
                        )
                    episode_seed += 1
                    observations, _ = env.reset(seed=episode_seed, episode_index=episode_index)
                    episode_returns = {agent: 0.0 for agent in agents}
                    if new_session:
                        hidden = model.initial_hidden(len(agents), device)
                    elif isinstance(hidden, tuple):
                        hidden = tuple(part.detach() for part in hidden)
                    elif hidden is not None:
                        hidden = hidden.detach()

                if global_step >= config.total_steps:
                    break

            with torch.no_grad():
                final_obs = np.stack([observations[a] for a in agents])
                final_state = np.repeat(
                    env.global_state(observations)[None, :], len(agents), axis=0
                )
                final_labels = torch.full(
                    (len(agents),), env.true_label(), dtype=torch.long, device=device
                )
                final_output = model.forward_step(
                    torch.as_tensor(final_obs, dtype=torch.float32, device=device),
                    torch.as_tensor(final_state, dtype=torch.float32, device=device),
                    hidden,
                    final_labels,
                    oracle_gate=model.selective,
                )
                next_values = final_output.values.cpu().numpy()

            rewards = np.asarray(buffer["rewards"])
            dones = np.asarray(buffer["dones"])
            values = np.asarray(buffer["values"])
            advantages = np.zeros_like(rewards)
            last_advantage = np.zeros(len(agents), dtype=np.float32)
            for step in reversed(range(len(rewards))):
                next_value = next_values if step == len(rewards) - 1 else values[step + 1]
                nonterminal = 1.0 - dones[step]
                delta = rewards[step] + config.gamma * next_value * nonterminal - values[step]
                last_advantage = (
                    delta + config.gamma * config.gae_lambda * nonterminal * last_advantage
                )
                advantages[step] = last_advantage
            returns = advantages + values

            def flat(values_: list) -> np.ndarray:
                array = np.asarray(values_)
                return array.reshape(-1, *array.shape[2:])

            obs_t = torch.as_tensor(flat(buffer["obs"]), dtype=torch.float32, device=device)
            states_t = torch.as_tensor(flat(buffer["states"]), dtype=torch.float32, device=device)
            actions_t = torch.as_tensor(flat(buffer["actions"]), dtype=torch.long, device=device)
            old_log_t = torch.as_tensor(
                flat(buffer["log_probs"]), dtype=torch.float32, device=device
            )
            returns_t = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
            advantages_t = torch.as_tensor(
                advantages.reshape(-1), dtype=torch.float32, device=device
            )
            labels_t = torch.as_tensor(flat(buffer["labels"]), dtype=torch.long, device=device)
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            hidden_a = None
            hidden_b = None
            if buffer["hidden_a"]:
                hidden_a = torch.as_tensor(
                    flat(buffer["hidden_a"]), dtype=torch.float32, device=device
                )
            if buffer["hidden_b"]:
                hidden_b = torch.as_tensor(
                    flat(buffer["hidden_b"]), dtype=torch.float32, device=device
                )

            indexes = np.arange(len(obs_t))
            losses: list[float] = []
            diagnosis_accuracies: list[float] = []
            for _ in range(config.update_epochs):
                np.random.shuffle(indexes)
                for start in range(0, len(indexes), config.minibatch_size):
                    batch = indexes[start : start + config.minibatch_size]
                    batch_hidden = None
                    if hidden_a is not None:
                        batch_hidden = hidden_a[batch]
                        if hidden_b is not None:
                            batch_hidden = (batch_hidden, hidden_b[batch])
                    log_prob, entropy, predicted_values, diagnosis = model.evaluate_actions(
                        obs_t[batch],
                        states_t[batch],
                        actions_t[batch],
                        batch_hidden,
                        labels_t[batch],
                    )
                    ratio = torch.exp(log_prob - old_log_t[batch])
                    policy_loss = -torch.min(
                        ratio * advantages_t[batch],
                        torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
                        * advantages_t[batch],
                    ).mean()
                    value_loss = F.mse_loss(predicted_values, returns_t[batch])
                    diagnosis_loss = torch.tensor(0.0, device=device)
                    if diagnosis is not None:
                        diagnosis_loss = F.cross_entropy(diagnosis, labels_t[batch])
                        diagnosis_accuracies.append(
                            float((diagnosis.argmax(-1) == labels_t[batch]).float().mean().item())
                        )
                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy.mean()
                        + config.diagnosis_coef * diagnosis_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    losses.append(float(loss.item()))

            update_index += 1
            metrics = {
                "update": update_index,
                "global_step": global_step,
                "mean_loss": float(np.mean(losses)),
                "recent_team_return": float(np.mean(completed_returns[-20:]))
                if completed_returns
                else None,
                "diagnosis_accuracy": float(np.mean(diagnosis_accuracies))
                if diagnosis_accuracies
                else None,
                "steps_per_second": global_step / max(time.perf_counter() - started, 1e-9),
            }
            artifacts.append_jsonl("train_metrics.jsonl", metrics)
            if update_index % config.log_interval == 0:
                print(
                    f"[{config.algorithm}] step={global_step}/{config.total_steps} "
                    f"return={metrics['recent_team_return']} loss={metrics['mean_loss']:.4f} "
                    f"diagnosis={metrics['diagnosis_accuracy']}"
                )

        checkpoint = {
            "algorithm": config.algorithm,
            "model_state": model.state_dict(),
            "config": asdict(config),
        }
        torch.save(checkpoint, artifacts.path / "checkpoint.pt")
        artifacts.write_json(
            "train_summary.json",
            {
                "episodes": len(completed_returns),
                "mean_last_20_return": float(np.mean(completed_returns[-20:])),
                "global_steps": global_step,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        artifacts.bundle()
        return artifacts.path
    finally:
        env.close()
