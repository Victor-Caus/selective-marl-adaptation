"""Discrete MADDPG baseline with centralized critics and decentralized actors."""

from __future__ import annotations

import copy
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from selective_marl.artifacts import RunArtifacts
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import (
    ACTION_COUNT,
    GLOBAL_STATE_SIZE,
    OBSERVATION_SIZE,
    ReferenceSessionEnv,
    make_session_spec,
)
from selective_marl.models import PolicyStep, mlp

if TYPE_CHECKING:
    from selective_marl.training import TrainConfig


class Transition(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class MADDPGPolicy(nn.Module):
    recurrent = False
    selective = False

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.actors = nn.ModuleList(
            [mlp(OBSERVATION_SIZE, hidden_size, ACTION_COUNT) for _ in range(2)]
        )
        critic_input = GLOBAL_STATE_SIZE + 2 * ACTION_COUNT
        self.critics = nn.ModuleList([mlp(critic_input, hidden_size, 1) for _ in range(2)])

    def initial_hidden(self, batch_size: int, device: torch.device) -> None:
        del batch_size, device
        return None

    def forward_step(
        self,
        observations: Tensor,
        states: Tensor,
        hidden=None,
        labels=None,
        *,
        deterministic=False,
        oracle_gate=False,
    ) -> PolicyStep:
        del hidden, labels, oracle_gate, states
        logits = torch.stack([self.actors[index](observations[index]) for index in range(2)])
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
        values = torch.zeros(2, device=observations.device)
        return PolicyStep(actions, log_probs, values, None, None)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_parameter, parameter in zip(target.parameters(), source.parameters(), strict=True):
        target_parameter.data.mul_(1 - tau).add_(parameter.data, alpha=tau)


def _joint_actions(actions: Tensor) -> Tensor:
    return F.one_hot(actions.long(), ACTION_COUNT).float().flatten(start_dim=1)


def train_maddpg(config: TrainConfig, output_root: Path) -> Path:
    from selective_marl.training import choose_device, set_seed

    set_seed(config.seed)
    device = choose_device(config.device)
    artifacts = RunArtifacts(output_root, f"train-maddpg-seed{config.seed}")
    artifacts.metadata()
    artifacts.write_json("config.json", asdict(config))
    model = MADDPGPolicy(config.hidden_size).to(device)
    targets = copy.deepcopy(model).to(device)
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.learning_rate) for actor in model.actors
    ]
    critic_optimizers = [
        torch.optim.Adam(critic.parameters(), lr=config.learning_rate) for critic in model.critics
    ]
    replay: deque[Transition] = deque(maxlen=100_000)
    batch_size = max(64, min(config.minibatch_size, 256))
    warmup = min(1_000, max(100, config.total_steps // 10))
    gamma = config.gamma
    tau = 0.01

    spec = make_session_spec(InterventionKind.NONE, config.change_episode, config.seed)
    env = ReferenceSessionEnv(spec, max_cycles=config.max_cycles)
    episode_index = 0
    session_index = 0
    episode_seed = config.seed * 100_000
    observations, _ = env.reset(seed=episode_seed, episode_index=episode_index)
    agents = env.possible_agents
    returns = {agent: 0.0 for agent in agents}
    completed: list[float] = []
    global_step = 0
    updates = 0
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    try:
        while global_step < config.total_steps:
            obs_array = np.stack([observations[a] for a in agents])
            epsilon = max(0.05, 1.0 - global_step / max(config.total_steps * 0.8, 1))
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
                greedy = (
                    torch.stack([model.actors[i](obs_tensor[i]).argmax() for i in range(2)])
                    .cpu()
                    .numpy()
                )
            actions_array = np.asarray(
                [
                    int(rng.integers(ACTION_COUNT)) if rng.random() < epsilon else int(greedy[i])
                    for i in range(2)
                ]
            )
            actions = {agent: int(actions_array[i]) for i, agent in enumerate(agents)}
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            reward_array = np.asarray([rewards[a] for a in agents], dtype=np.float32)
            done_array = np.asarray(
                [terminations.get(a, False) or truncations.get(a, False) for a in agents],
                dtype=np.float32,
            )
            next_array = (
                np.stack([next_observations[a] for a in agents])
                if next_observations
                else np.zeros_like(obs_array)
            )
            replay.append(
                Transition(obs_array, actions_array, reward_array, next_array, done_array)
            )
            for agent in agents:
                returns[agent] += float(rewards[agent])
            observations = next_observations
            global_step += 2

            if len(replay) >= warmup:
                sample = random.sample(replay, min(batch_size, len(replay)))
                obs = torch.as_tensor(
                    np.stack([item.observations for item in sample]),
                    dtype=torch.float32,
                    device=device,
                )
                acts = torch.as_tensor(np.stack([item.actions for item in sample]), device=device)
                rewards_t = torch.as_tensor(
                    np.stack([item.rewards for item in sample]), device=device
                )
                next_obs = torch.as_tensor(
                    np.stack([item.next_observations for item in sample]),
                    dtype=torch.float32,
                    device=device,
                )
                dones_t = torch.as_tensor(np.stack([item.dones for item in sample]), device=device)
                state = obs.flatten(start_dim=1)
                next_state = next_obs.flatten(start_dim=1)
                joint = _joint_actions(acts)
                with torch.no_grad():
                    next_actions = torch.stack(
                        [targets.actors[i](next_obs[:, i]).argmax(-1) for i in range(2)], dim=1
                    )
                    next_joint = _joint_actions(next_actions)
                critic_losses = []
                for index in range(2):
                    with torch.no_grad():
                        target_q = rewards_t[:, index] + gamma * (1 - dones_t[:, index]) * (
                            targets.critics[index](
                                torch.cat([next_state, next_joint], dim=-1)
                            ).squeeze(-1)
                        )
                    predicted_q = model.critics[index](torch.cat([state, joint], dim=-1)).squeeze(
                        -1
                    )
                    critic_loss = F.mse_loss(predicted_q, target_q)
                    critic_optimizers[index].zero_grad()
                    critic_loss.backward()
                    critic_optimizers[index].step()
                    critic_losses.append(float(critic_loss.item()))

                    policy_actions = F.one_hot(acts, ACTION_COUNT).float()
                    policy_actions[:, index] = F.softmax(model.actors[index](obs[:, index]), dim=-1)
                    actor_loss = -model.critics[index](
                        torch.cat([state, policy_actions.flatten(start_dim=1)], dim=-1)
                    ).mean()
                    actor_optimizers[index].zero_grad()
                    actor_loss.backward()
                    actor_optimizers[index].step()
                    _soft_update(targets.actors[index], model.actors[index], tau)
                    _soft_update(targets.critics[index], model.critics[index], tau)
                updates += 1
                if updates % max(1, config.rollout_steps // 8) == 0:
                    artifacts.append_jsonl(
                        "train_metrics.jsonl",
                        {
                            "global_step": global_step,
                            "updates": updates,
                            "critic_loss": float(np.mean(critic_losses)),
                            "epsilon": epsilon,
                            "recent_team_return": float(np.mean(completed[-20:]))
                            if completed
                            else None,
                        },
                    )

            if bool(np.all(done_array)):
                team_return = float(np.mean(list(returns.values())))
                completed.append(team_return)
                artifacts.append_jsonl(
                    "episodes.jsonl",
                    {
                        "global_step": global_step,
                        "session": session_index,
                        "episode": episode_index,
                        "condition": "none",
                        "team_return": team_return,
                    },
                )
                episode_index += 1
                if episode_index >= config.session_episodes:
                    session_index += 1
                    episode_index = 0
                episode_seed += 1
                observations, _ = env.reset(seed=episode_seed, episode_index=episode_index)
                returns = {agent: 0.0 for agent in agents}
                if len(completed) % 10 == 0:
                    print(
                        f"[maddpg] step={global_step}/{config.total_steps} "
                        f"return={np.mean(completed[-20:]):.3f} epsilon={epsilon:.3f}"
                    )

        torch.save(
            {
                "algorithm": "maddpg",
                "model_state": model.state_dict(),
                "config": asdict(config),
            },
            artifacts.path / "checkpoint.pt",
        )
        artifacts.write_json(
            "train_summary.json",
            {
                "episodes": len(completed),
                "mean_last_20_return": float(np.mean(completed[-20:])),
                "global_steps": global_step,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        artifacts.bundle()
        return artifacts.path
    finally:
        env.close()
