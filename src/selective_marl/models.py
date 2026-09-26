"""Shared-actor centralized-critic policies used by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from selective_marl.environments.reference import (
    ACTION_COUNT,
    GLOBAL_STATE_SIZE,
    MESSAGE_COUNT,
    OBSERVATION_SIZE,
    PHYSICAL_OBSERVATION_SIZE,
)


def mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_size),
    )


@dataclass(slots=True)
class PolicyStep:
    actions: Tensor
    log_probs: Tensor
    values: Tensor
    hidden: Tensor | tuple[Tensor, Tensor] | None
    diagnosis_logits: Tensor | None


class MAPPOPolicy(nn.Module):
    recurrent = False
    selective = False

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.actor = mlp(OBSERVATION_SIZE, hidden_size, ACTION_COUNT)
        self.critic = mlp(GLOBAL_STATE_SIZE, hidden_size, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> None:
        return None

    def forward_step(
        self,
        observations: Tensor,
        states: Tensor,
        hidden: Tensor | tuple[Tensor, Tensor] | None = None,
        labels: Tensor | None = None,
        *,
        deterministic: bool = False,
        oracle_gate: bool = False,
    ) -> PolicyStep:
        del hidden, labels, oracle_gate
        logits = self.actor(observations)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return PolicyStep(
            actions,
            distribution.log_prob(actions),
            self.critic(states).squeeze(-1),
            None,
            None,
        )

    def evaluate_actions(
        self,
        observations: Tensor,
        states: Tensor,
        actions: Tensor,
        hidden: Tensor | tuple[Tensor, Tensor] | None,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        del hidden, labels
        distribution = Categorical(logits=self.actor(observations))
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(states).squeeze(-1),
            None,
        )


class RecurrentMAPPOPolicy(MAPPOPolicy):
    recurrent = True

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__(hidden_size)
        self.encoder = nn.Sequential(nn.Linear(OBSERVATION_SIZE, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor_head = nn.Linear(hidden_size, ACTION_COUNT)

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _features(self, observations: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        next_hidden = self.gru(self.encoder(observations), hidden)
        return next_hidden, next_hidden

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
        del labels, oracle_gate
        if hidden is None:
            hidden = self.initial_hidden(len(observations), observations.device)
        features, next_hidden = self._features(observations, hidden)
        logits = self.actor_head(features)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return PolicyStep(
            actions,
            distribution.log_prob(actions),
            self.critic(states).squeeze(-1),
            next_hidden,
            None,
        )

    def evaluate_actions(self, observations, states, actions, hidden, labels):
        del labels
        if hidden is None:
            hidden = self.initial_hidden(len(observations), observations.device)
        features, _ = self._features(observations, hidden)
        distribution = Categorical(logits=self.actor_head(features))
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(states).squeeze(-1),
            None,
        )


class SelectiveMAPPOPolicy(MAPPOPolicy):
    recurrent = True
    selective = True

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__(hidden_size)
        half = hidden_size // 2
        self.context_size = half
        self.physical_encoder = nn.Sequential(nn.Linear(PHYSICAL_OBSERVATION_SIZE, half), nn.Tanh())
        self.semantic_encoder = nn.Sequential(nn.Linear(MESSAGE_COUNT, half), nn.Tanh())
        self.behavior_gru = nn.GRUCell(half, half)
        self.semantic_gru = nn.GRUCell(half, half)
        self.diagnoser = mlp(hidden_size * 2, hidden_size, 4)
        self.actor_head = mlp(hidden_size, hidden_size, ACTION_COUNT)

    def initial_hidden(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        zeros = torch.zeros(batch_size, self.context_size, device=device)
        return zeros.clone(), zeros.clone()

    def _features(
        self,
        observations: Tensor,
        hidden: tuple[Tensor, Tensor],
        labels: Tensor | None,
        oracle_gate: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        physical = self.physical_encoder(observations[:, :PHYSICAL_OBSERVATION_SIZE])
        semantic = self.semantic_encoder(observations[:, -MESSAGE_COUNT:])
        behavior_hidden, semantic_hidden = hidden
        behavior_candidate = self.behavior_gru(physical, behavior_hidden)
        semantic_candidate = self.semantic_gru(semantic, semantic_hidden)
        logits = self.diagnoser(
            torch.cat([physical, semantic, behavior_hidden, semantic_hidden], dim=-1)
        )
        gate_labels = labels if oracle_gate and labels is not None else logits.argmax(dim=-1)
        # Before a detected shift (class 0), both contexts track the normal regime.
        update_behavior = ((gate_labels == 0) | (gate_labels == 2) | (gate_labels == 3)).float()
        update_semantic = ((gate_labels == 0) | (gate_labels == 1) | (gate_labels == 3)).float()
        next_behavior = (
            update_behavior[:, None] * behavior_candidate
            + (1 - update_behavior[:, None]) * behavior_hidden
        )
        next_semantic = (
            update_semantic[:, None] * semantic_candidate
            + (1 - update_semantic[:, None]) * semantic_hidden
        )
        return (
            torch.cat([next_behavior, next_semantic], dim=-1),
            (next_behavior, next_semantic),
            logits,
        )

    def forward_step(
        self,
        observations,
        states,
        hidden=None,
        labels=None,
        *,
        deterministic=False,
        oracle_gate=False,
    ) -> PolicyStep:
        if hidden is None:
            hidden = self.initial_hidden(len(observations), observations.device)
        features, next_hidden, diagnosis = self._features(observations, hidden, labels, oracle_gate)
        logits = self.actor_head(features)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return PolicyStep(
            actions,
            distribution.log_prob(actions),
            self.critic(states).squeeze(-1),
            next_hidden,
            diagnosis,
        )

    def evaluate_actions(self, observations, states, actions, hidden, labels):
        if hidden is None:
            hidden = self.initial_hidden(len(observations), observations.device)
        features, _, diagnosis = self._features(observations, hidden, labels, True)
        distribution = Categorical(logits=self.actor_head(features))
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(states).squeeze(-1),
            diagnosis,
        )


def build_policy(algorithm: str, hidden_size: int = 128) -> MAPPOPolicy:
    if algorithm in {"mappo", "mappo_no_comm", "channel_randomized"}:
        return MAPPOPolicy(hidden_size)
    if algorithm == "rmappo":
        return RecurrentMAPPOPolicy(hidden_size)
    if algorithm == "selective":
        return SelectiveMAPPOPolicy(hidden_size)
    raise ValueError(f"unsupported algorithm: {algorithm}")
