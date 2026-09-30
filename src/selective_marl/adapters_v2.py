"""Supervised semantic memory and conservative label-free runtime routing.

Simulator labels are training targets only. Runtime inputs are local observations,
previous own actions/rewards, and an episode-reset bit. V1 remains untouched.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from selective_marl.adapters import OnlineAdapters

V2_MODES = {
    "selective_v2",
    "joint_v2",
    "motor_only",
    "unguarded_v2",
    "reward_only_v2",
    "confidence_only_v2",
    "no_rollback_v2",
    "shared_stream_v2",
}


def memory_features(obs, previous_actions, previous_rewards, reset):
    actions = np.asarray(previous_actions, dtype=int)
    return np.concatenate(
        [
            np.asarray(obs, dtype=np.float32),
            np.eye(5)[actions[:, 0]],
            np.eye(10)[actions[:, 1]],
            np.asarray(previous_rewards)[:, None] / 5.0,
            np.full((len(actions), 1), float(reset)),
        ],
        axis=-1,
    ).astype(np.float32)


class SemanticMemory(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(38, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, 10)
        self.diagnoser = nn.Linear(hidden_size, 1)

    def forward(self, sequence, hidden=None):
        encoded, hidden = self.gru(self.encoder(sequence), hidden)
        return self.decoder(encoded), self.diagnoser(encoded).squeeze(-1), hidden


class ConservativeDetector:
    def __init__(self, threshold=6.0, warmup=8):
        self.threshold = threshold
        self.warmup = warmup
        self.history = []
        self.cusum = np.zeros(2)
        self.bad_streak = np.zeros(2, dtype=int)

    def observe(self, returns):
        returns = np.asarray(returns, dtype=float)
        signal = np.zeros(2, dtype=bool)
        if len(self.history) >= self.warmup:
            baseline = np.asarray(self.history[: self.warmup])
            center = np.median(baseline, axis=0)
            scale = np.maximum(1.4826 * np.median(abs(baseline - center), axis=0), 3.0)
            z = (center - returns) / scale
            self.cusum = np.maximum(0, self.cusum + z - 0.5)
            self.bad_streak = np.where(z > 0.5, self.bad_streak + 1, 0)
            signal = (self.cusum >= self.threshold) & (self.bad_streak >= 3)
        self.history.append(returns.copy())
        return signal


class GuardedAdapters(OnlineAdapters):
    """Frozen sender stream, learned receiver memory, and retained motor identification."""

    def __init__(self, actor, memory=None, mode="selective_v2", threshold=6.0):
        super().__init__(actor, mode="frozen" if mode == "frozen" else "selective")
        self.v2_mode = mode
        self.memory = None if mode == "motor_only" else memory
        if self.memory is not None:
            self.memory.eval()
            self.memory.requires_grad_(False)
        self.detector = ConservativeDetector(threshold)
        self.context = None
        self.previous_actions = np.zeros((2, 2), dtype=int)
        self.previous_rewards = np.zeros(2)
        self.episode_reset = True
        self.sender_hidden = None
        self.probabilities = []
        self.cooldown = np.zeros(2, dtype=int)
        self.trials = [None, None]
        self.rollbacks = np.zeros(2, dtype=int)

    def begin_episode(self):
        self.episode_reset = True
        self.previous_actions.fill(0)
        self.previous_rewards.fill(0)
        self.sender_hidden = None
        self.probabilities = []

    def distributions(self, obs, hidden):
        raw = torch.as_tensor(obs, dtype=torch.float32)
        translated = raw.clone()
        probability = np.zeros(2)
        if self.memory is not None:
            features = memory_features(
                obs, self.previous_actions, self.previous_rewards, self.episode_reset
            )
            decoded, diagnosis, self.context = self.memory(
                torch.from_numpy(features)[None], self.context
            )
            probability = diagnosis[0].sigmoid().detach().numpy()
            recovered = decoded[0].softmax(-1)
            # Preserve the no-message vector at the start of every episode.
            recovered = recovered * raw[:, -10:].sum(-1, keepdim=True)
            gate = torch.from_numpy(self.semantic_active.astype(np.float32))[:, None]
            translated[:, -10:] = gate * recovered + (1 - gate) * raw[:, -10:]
        self.probabilities.append(probability)
        self.episode_reset = False
        masks = torch.ones(2, 1)
        features, hidden = self.actor.rnn(self.actor.base(translated), hidden, masks)
        move = self.actor.act.action_outs[0](features).probs
        move = torch.bmm(move[:, None, :], self.motor).squeeze(1)
        # Independent hidden state: receiver adaptations cannot alter sent symbols.
        if self.sender_hidden is None:
            self.sender_hidden = torch.zeros_like(hidden)
        sender, self.sender_hidden = self.actor.rnn(self.actor.base(raw), self.sender_hidden, masks)
        message = self.actor.act.action_outs[1](sender).probs
        if self.v2_mode == "shared_stream_v2":
            # Matched ablation: same weights and gate, but both action heads use
            # the receiver-adapted recurrent stream (as in the base actor).
            message = self.actor.act.action_outs[1](features).probs
        return Categorical(probs=move), Categorical(probs=message), hidden

    def feedback(self, obs, actions, rewards, next_obs):
        self.observe_transition(obs, actions, next_obs)
        self.previous_actions = np.asarray(actions).copy()
        self.previous_rewards = np.asarray(rewards).copy()

    def observe_return(self, returns):
        returns = np.asarray(returns)
        signals = self.detector.observe(returns)
        self.cooldown = np.maximum(0, self.cooldown - 1)
        confidence = np.mean(self.probabilities, axis=0) if self.probabilities else np.zeros(2)
        for agent in range(2):
            trial = self.trials[agent]
            if trial is not None:
                trial["after"].append(float(returns[agent]))
                if len(trial["after"]) >= 4:
                    before, after = np.asarray(trial["before"]), np.asarray(trial["after"])
                    margin = max(
                        3.0, 2 * np.sqrt(before.var() / len(before) + after.var() / len(after))
                    )
                    if after.mean() < before.mean() - margin:
                        self.semantic_active[agent] = False
                        self.cooldown[agent] = 8
                        self.rollbacks[agent] += 1
                        self.detector.cusum[agent] = 0
                    self.trials[agent] = None
            activate = signals[agent] and confidence[agent] >= 0.8
            if self.v2_mode == "reward_only_v2":
                activate = signals[agent]
            if self.v2_mode == "confidence_only_v2":
                activate = (
                    len(self.detector.history) >= self.detector.warmup and confidence[agent] >= 0.8
                )
            if self.v2_mode == "joint_v2":
                activate = activate or self.motor_active[agent]
            if self.v2_mode == "unguarded_v2":
                activate = len(self.detector.history) >= self.detector.warmup
            if (
                self.v2_mode != "frozen"
                and self.memory is not None
                and activate
                and not self.semantic_active[agent]
                and self.cooldown[agent] == 0
            ):
                self.semantic_active[agent] = True
                if self.v2_mode not in {"unguarded_v2", "no_rollback_v2"}:
                    self.trials[agent] = {
                        "before": [r[agent] for r in self.detector.history[-4:]],
                        "after": [],
                    }
        self.return_history.append(returns.tolist())

    def update(self, trajectory, initial_hidden, epochs=4):
        # V2 memory is trained offline on training-only interventions. At test time
        # its recurrent context adapts; no target labels or PPO gradients are used.
        return
