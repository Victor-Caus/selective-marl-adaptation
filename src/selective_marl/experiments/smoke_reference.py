"""Random-policy smoke test for the installed MPE2 Simple Reference environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
from mpe2 import simple_reference_v3


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    environment: str
    episodes: int
    seed: int
    mean_team_return: float
    std_team_return: float
    agents: tuple[str, ...]


def run(episodes: int, seed: int) -> SmokeSummary:
    if episodes < 1:
        raise ValueError("episodes must be positive")

    env = simple_reference_v3.parallel_env(continuous_actions=False)
    rng = np.random.default_rng(seed)
    team_returns: list[float] = []
    agent_names: tuple[str, ...] = ()

    try:
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode)
            agent_names = tuple(env.agents)
            totals = {agent: 0.0 for agent in env.agents}
            while env.agents:
                actions = {
                    agent: int(rng.integers(env.action_space(agent).n)) for agent in env.agents
                }
                observations, rewards, terminations, truncations, _ = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                if observations and not all(
                    terminations.get(agent, False) or truncations.get(agent, False)
                    for agent in terminations
                ):
                    continue
            team_returns.append(float(np.mean(list(totals.values()))))
    finally:
        env.close()

    return SmokeSummary(
        environment="simple_reference_v3",
        episodes=episodes,
        seed=seed,
        mean_team_return=float(np.mean(team_returns)),
        std_team_return=float(np.std(team_returns)),
        agents=agent_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(asdict(run(args.episodes, args.seed)), indent=2))


if __name__ == "__main__":
    main()

