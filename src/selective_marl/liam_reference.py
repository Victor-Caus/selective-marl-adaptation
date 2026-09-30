"""Dimension-only bridge to the MIT-licensed official LIAM implementation.

The external optimizer, losses, recurrence, return normalization and rollout
methods are imported without edits. MPE2 has 21 observations and 5+10 actions,
whereas the original double-speaker task used 18 observations and 5+5 actions.
"""

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[2] / ".external/reference-sources/liam"
COMMIT = "8545b9e4237eb60ad45b7cb8ed6caec6bc4263b5"


@lru_cache(maxsize=1)
def official():
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != COMMIT:
        raise RuntimeError(f"Unpinned LIAM revision: {revision}")
    if subprocess.check_output(
        ["git", "-C", str(SOURCE), "diff", "--", "double_speaker_listener"], text=True
    ):
        raise RuntimeError("Official LIAM source has local modifications")
    folder = SOURCE / "double_speaker_listener"
    sys.path.insert(0, str(folder))
    from agent import A2C
    from standardise_stream import RunningMeanStd
    from storage import RolloutStorage

    return A2C, RolloutStorage, RunningMeanStd


def make_agent(obs=21, communication=10):
    A2C, _, _ = official()
    agent = A2C(obs, 128, 20, 5, obs, 5, 3e-4, 7e-4, 0.01, 0.5, 0.5)
    if communication != 5:
        agent.actor_critic.policy2 = torch.nn.Linear(128, communication)
        agent.encoder.lstm = torch.nn.LSTM(obs + 5 + communication, 128)
        agent.decoder.fc5 = torch.nn.Linear(128, communication)
        agent.optimizer1 = torch.optim.Adam(agent.actor_critic.parameters(), lr=3e-4)
        agent.optimizer2 = torch.optim.Adam(
            list(agent.encoder.parameters()) + list(agent.decoder.parameters()), lr=7e-4
        )
    return agent


def make_rollout(count, steps=5, communication=10):
    _, Storage, _ = official()
    rollout = Storage(steps, count, 21, 5, 128, 21, 5 + communication)
    rollout.actions2 = torch.zeros(steps + 1, count, communication)
    return rollout


def hidden(count):
    return (torch.zeros(1, count, 128), torch.zeros(1, count, 128))
