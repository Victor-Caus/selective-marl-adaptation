"""Exploratory online adapters; receive only local observations/actions/rewards.

The frozen policy supplies communication and navigation. Two small stochastic
matrices translate incoming symbols and outgoing movements. PPO updates these
matrices without changing the pretrained policy. Routing uses action/velocity
prediction errors and return degradation, never simulator intervention labels.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.distributions import Categorical


class OnlineAdapters(nn.Module):
    def __init__(self, actor, mode="selective", learning_rate=0.03):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.actor.requires_grad_(False)
        self.semantic = nn.Parameter(torch.eye(10).repeat(2, 1, 1))
        self.motor = nn.Parameter(torch.eye(5).repeat(2, 1, 1))
        self.optimizer = torch.optim.Adam([self.semantic, self.motor], lr=learning_rate)
        self.mode = mode
        self.return_history = []
        self.semantic_active = np.zeros(2, dtype=bool)
        self.motor_active = np.zeros(2, dtype=bool)
        self.velocity_data = [[], []]
        self.dynamics = [None, None]
        self.motor_evidence = np.zeros(2, dtype=int)
        self.motor_counts = np.eye(5)[None].repeat(2, axis=0) * 0.1

    def distributions(self, obs, hidden):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        message = torch.bmm(obs[:, None, -10:], self.semantic).squeeze(1)
        translated = torch.cat([obs[:, :11], message], dim=-1)
        features = self.actor.base(translated)
        features, hidden = self.actor.rnn(features, hidden, torch.ones(2, 1))
        move = self.actor.act.action_outs[0](features).probs
        move = torch.bmm(move[:, None, :], self.motor).squeeze(1)
        message_dist = self.actor.act.action_outs[1](features)
        return Categorical(probs=move), Categorical(probs=message_dist.probs), hidden

    def observe_transition(self, obs, actions, next_obs):
        # Learn nominal local velocity dynamics from the first 100 transitions.
        # No absolute episode number, change point, permutation or label is used.
        for agent in range(2):
            x = np.r_[obs[agent, :2], np.eye(5)[actions[agent, 0]]]
            y = next_obs[agent, :2]
            if self.dynamics[agent] is None:
                self.velocity_data[agent].append((x, y))
                if len(self.velocity_data[agent]) >= 100:
                    xs, ys = zip(*self.velocity_data[agent], strict=True)
                    self.dynamics[agent] = np.linalg.lstsq(np.asarray(xs), ys, rcond=None)[0]
            else:
                error = np.linalg.norm(x @ self.dynamics[agent] - y)
                candidates = (
                    np.concatenate([np.repeat(obs[agent, :2][None], 5, axis=0), np.eye(5)], axis=1)
                    @ self.dynamics[agent]
                )
                closest = int(np.linalg.norm(candidates - y, axis=1).argmin())
                command = actions[agent, 0]
                self.motor_counts[agent, command] *= 0.5
                self.motor_counts[agent, command, closest] += 1
                self.motor_evidence[agent] = self.motor_evidence[agent] + 1 if error > 0.05 else 0
                if self.motor_evidence[agent] >= 3:
                    self.motor_active[agent] = True
                if self.motor_active[agent] and self.mode != "frozen":
                    # Generic learned action remapping, not the harness's rotation.
                    commands, effects = linear_sum_assignment(-self.motor_counts[agent])
                    inverse = np.zeros((5, 5), dtype=np.float32)
                    inverse[effects, commands] = 1
                    with torch.no_grad():
                        self.motor[agent].copy_(torch.from_numpy(inverse))

    def observe_return(self, returns):
        returns = np.asarray(returns)
        if len(self.return_history) >= 8:
            baseline = np.asarray(self.return_history[:8])
            threshold = baseline.mean(0) - np.maximum(2 * baseline.std(0), 3.0)
            # Require two bad episodes. Behavioral evidence is a separate route.
            previous = np.asarray(self.return_history[-1])
            bad = (returns < threshold) & (previous < threshold)
            self.semantic_active |= bad & ~self.motor_active
            # Once movement has an adapter, residual reward degradation can also
            # activate semantics; the joint shift need not be mutually exclusive.
            corrected = (self.motor.detach().numpy() - np.eye(5)).max((1, 2)) > 0.2
            self.semantic_active |= bad & corrected
        self.return_history.append(returns.tolist())

    def update(self, trajectory, initial_hidden, epochs=4):
        if self.mode == "frozen":
            return
        # The joint control uses the SAME detector and starts at the same time;
        # its only difference is updating both components after any detected shift.
        detected = self.semantic_active | self.motor_active
        sem_gate = detected if self.mode == "joint" else self.semantic_active
        mot_gate = detected if self.mode == "joint" else self.motor_active
        # Once identified from observed transitions, preserve the dynamics solution.
        mot_gate = mot_gate & ~self.motor_active
        if not sem_gate.any() and not mot_gate.any():
            return
        obs, actions, old_log, rewards = trajectory
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        old_log = torch.as_tensor(np.asarray(old_log), dtype=torch.float32)
        discounted = np.zeros_like(rewards, dtype=np.float32)
        carry = np.zeros(2, dtype=np.float32)
        for t in reversed(range(len(rewards))):
            carry = np.asarray(rewards[t]) + 0.99 * carry
            discounted[t] = carry
        advantage = torch.as_tensor(discounted)
        advantage = (advantage - advantage.mean(0)) / (advantage.std(0) + 1e-5)
        for _ in range(epochs):
            hidden = initial_hidden.clone()
            logs, entropies = [], []
            for t in range(len(obs)):
                move, message, hidden = self.distributions(obs[t], hidden)
                logs.append(move.log_prob(actions[t, :, 0]) + message.log_prob(actions[t, :, 1]))
                entropies.append(move.entropy() + message.entropy())
            ratio = (torch.stack(logs) - old_log).exp()
            loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            loss -= 0.01 * torch.stack(entropies).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.semantic.grad *= torch.as_tensor(sem_gate)[:, None, None]
            self.motor.grad *= torch.as_tensor(mot_gate)[:, None, None]
            # Restore gated slices as well: Adam momentum must not update a frozen component.
            sem_before, mot_before = self.semantic.detach().clone(), self.motor.detach().clone()
            nn.utils.clip_grad_norm_([self.semantic, self.motor], 0.5)
            self.optimizer.step()
            with torch.no_grad():
                for matrix in (self.semantic, self.motor):
                    matrix.clamp_(min=0)
                    matrix.div_(matrix.sum(-1, keepdim=True).clamp_min(1e-8))
                self.semantic[~sem_gate] = sem_before[~sem_gate]
                self.motor[~mot_gate] = mot_before[~mot_gate]
