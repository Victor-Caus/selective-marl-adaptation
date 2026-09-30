"""MPE2 adapter for the pinned on-policy recurrent trainer.

Action indices retain the historical *one-hot* MPE convention. That convention
has the opposite signs to MPE2 discrete actions; explicitly translate both axes.
Rewards sum the two MPE2 distances. Only the aggregation matches historical MPE;
the historical task uses squared distance and a different landmark distribution.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box
from onpolicy.envs.mpe.multi_discrete import MultiDiscrete

from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec

HISTORICAL_TO_MPE2 = np.asarray([0, 2, 1, 4, 3])


class ReferenceBackendEnv:
    def __init__(self, seed: int = 1, horizon: int = 25):
        self.env = ReferenceSessionEnv(
            make_session_spec(InterventionKind.NONE, 0, seed),
            max_cycles=horizon,
            local_ratio=0.0,
        )
        self.n = 2
        self.observation_space = [Box(-np.inf, np.inf, (21,), np.float32) for _ in range(2)]
        self.share_observation_space = [Box(-np.inf, np.inf, (42,), np.float32) for _ in range(2)]
        self.action_space = [MultiDiscrete([[0, 4], [0, 9]]) for _ in range(2)]
        self.seed(seed)

    def seed(self, seed):
        self.next_seed = int(seed)

    def reset(self):
        observations, _ = self.env.reset(seed=self.next_seed, episode_index=0)
        self.next_seed += 1
        return np.stack([observations[a] for a in self.env.possible_agents])

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (2, 15):
            raise ValueError(f"Expected two concatenated one-hot actions, got {actions.shape}")
        movements = HISTORICAL_TO_MPE2[actions[:, :5].argmax(-1)]
        messages = actions[:, 5:].argmax(-1)
        encoded = movements + 5 * messages
        obs, rewards, terminated, truncated, infos = self.env.step(
            dict(zip(self.env.possible_agents, encoded.tolist(), strict=True))
        )
        names = self.env.possible_agents
        return (
            np.stack([obs[a] for a in names]),
            np.asarray([[2 * rewards[a]] for a in names], dtype=np.float32),
            np.asarray([terminated[a] or truncated[a] for a in names]),
            [infos[a] for a in names],
        )

    def close(self):
        self.env.close()


class RandomizedReferenceBackendEnv(ReferenceBackendEnv):
    """Training-only randomization with the V2 session distribution.

    Neither condition nor change point is added to policy observations. Original
    recurrent MAPPO resets its hidden state each episode, as in base training.
    """

    def __init__(self, seed=1, horizon=25):
        super().__init__(seed, horizon)
        self.rng = np.random.default_rng(seed)
        self.training_episode = 0
        # Training factories seed ranks at base_seed + 1000 * rank. Offset the
        # balanced cycle so short runs expose all four conditions from the start.
        self.training_session = (seed // 1000) % 4

    def reset(self):
        episode = self.training_episode % 48
        if episode == 0:
            condition = ("none", "semantic", "behavioral", "both")[self.training_session % 4]
            self.env.spec = make_session_spec(
                InterventionKind(condition),
                int(self.rng.integers(10, 31)),
                int(self.rng.integers(1, 2**31)),
            )
            self.training_session += 1
        observations, _ = self.env.reset(seed=self.next_seed, episode_index=episode)
        self.next_seed += 1
        self.training_episode += 1
        return np.stack([observations[a] for a in self.env.possible_agents])
