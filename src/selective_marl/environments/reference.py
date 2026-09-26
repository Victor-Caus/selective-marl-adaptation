"""Controlled Simple Reference sessions.

The wrapper intervenes on either the received symbol vector or the physical component of
one agent's discrete action. The other component is preserved exactly, which makes the
four intervention labels causal rather than post-hoc descriptions of a failed episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from mpe2 import simple_reference_v3

from selective_marl.environments.interventions import InterventionKind

MOVEMENT_COUNT = 5
MESSAGE_COUNT = 10
OBSERVATION_SIZE = 21
PHYSICAL_OBSERVATION_SIZE = 11
GLOBAL_STATE_SIZE = 42
ACTION_COUNT = MOVEMENT_COUNT * MESSAGE_COUNT


def decode_action(action: int) -> tuple[int, int]:
    """Decode MPE2's discrete product action into movement and message indexes."""

    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"action must be in [0, {ACTION_COUNT})")
    return action % MOVEMENT_COUNT, action // MOVEMENT_COUNT


def encode_action(movement: int, message: int) -> int:
    if not 0 <= movement < MOVEMENT_COUNT:
        raise ValueError("invalid movement index")
    if not 0 <= message < MESSAGE_COUNT:
        raise ValueError("invalid message index")
    return message * MOVEMENT_COUNT + movement


def rotate_movement(movement: int) -> int:
    """Rotate cardinal movement clockwise while keeping no-op unchanged."""

    return (0, 3, 4, 2, 1)[movement]


@dataclass(frozen=True, slots=True)
class SessionSpec:
    condition: InterventionKind
    change_episode: int
    semantic_permutation: tuple[int, ...]
    behavior_agent: str = "agent_1"

    def active(self, episode_index: int) -> bool:
        return episode_index >= self.change_episode

    def label(self, episode_index: int) -> InterventionKind:
        if not self.active(episode_index):
            return InterventionKind.NONE
        return self.condition


def make_session_spec(
    condition: InterventionKind,
    change_episode: int,
    seed: int,
) -> SessionSpec:
    rng = np.random.default_rng(seed)
    permutation = tuple(int(value) for value in rng.permutation(MESSAGE_COUNT))
    if permutation == tuple(range(MESSAGE_COUNT)):
        permutation = tuple(range(1, MESSAGE_COUNT)) + (0,)
    return SessionSpec(condition, change_episode, permutation)


class ReferenceSessionEnv:
    """Parallel MPE2 environment with mechanism-isolated interventions."""

    def __init__(
        self,
        spec: SessionSpec,
        *,
        max_cycles: int = 25,
        local_ratio: float = 0.5,
        render_mode: str | None = None,
        mask_messages: bool = False,
    ) -> None:
        self.spec = spec
        self.episode_index = 0
        self.mask_messages = mask_messages
        self.env = simple_reference_v3.parallel_env(
            max_cycles=max_cycles,
            local_ratio=local_ratio,
            continuous_actions=False,
            render_mode=render_mode,
        )

    @property
    def agents(self) -> list[str]:
        return self.env.agents

    @property
    def possible_agents(self) -> list[str]:
        return self.env.possible_agents

    def action_space(self, agent: str) -> Any:
        return self.env.action_space(agent)

    def reset(self, *, seed: int, episode_index: int) -> tuple[dict[str, np.ndarray], dict]:
        self.episode_index = episode_index
        observations, infos = self.env.reset(seed=seed)
        return self._observations(observations), infos

    def step(self, actions: dict[str, int]) -> tuple:
        transformed = dict(actions)
        if self._behavior_active() and self.spec.behavior_agent in transformed:
            movement, message = decode_action(int(transformed[self.spec.behavior_agent]))
            transformed[self.spec.behavior_agent] = encode_action(
                rotate_movement(movement), message
            )
        observations, rewards, terminations, truncations, infos = self.env.step(transformed)
        return self._observations(observations), rewards, terminations, truncations, infos

    def render(self) -> np.ndarray | None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def global_state(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([observations[name] for name in self.possible_agents]).astype(
            np.float32
        )

    def true_label(self) -> int:
        return list(InterventionKind).index(self.spec.label(self.episode_index))

    def _semantic_active(self) -> bool:
        return self.spec.active(self.episode_index) and self.spec.condition in {
            InterventionKind.SEMANTIC,
            InterventionKind.BOTH,
        }

    def _behavior_active(self) -> bool:
        return self.spec.active(self.episode_index) and self.spec.condition in {
            InterventionKind.BEHAVIORAL,
            InterventionKind.BOTH,
        }

    def _observations(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        permutation = np.asarray(self.spec.semantic_permutation)
        for agent, observation in observations.items():
            changed = np.asarray(observation, dtype=np.float32).copy()
            if self.mask_messages:
                changed[-MESSAGE_COUNT:] = 0.0
            elif self._semantic_active():
                changed[-MESSAGE_COUNT:] = changed[-MESSAGE_COUNT:][permutation]
            result[agent] = changed
        return result
