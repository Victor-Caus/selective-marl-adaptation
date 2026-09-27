"""Run pinned MAPPO with the official vector batch in one Windows process.

Windows uses the ``spawn`` multiprocessing method, so every official SubprocVecEnv worker
imports a private copy of PyTorch, SciPy, pandas, matplotlib, and wandb. On memory-constrained
hosts this fails before the first environment step. This adapter replaces only environment
process placement: it creates the same ranked and seeded environments in DummyVecEnv. The
official training loop, policy, optimizer, batch size, and update count remain unchanged.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

# The PowerShell runner starts this adapter from <short-junction>/onpolicy/scripts.
# Prefer that checkout before importing the editable installation at its long OneDrive path.
working_directory = Path.cwd()
if working_directory.name == "scripts" and working_directory.parent.name == "onpolicy":
    sys.path.insert(0, str(working_directory.parents[1]))

from onpolicy.envs.env_wrappers import DummyVecEnv
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.scripts.train import train_mpe


def make_sequential_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "MPE":
                raise NotImplementedError(f"Unsupported environment: {all_args.env_name}")
            env = MPEEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(rank) for rank in range(all_args.n_rollout_threads)])


if __name__ == "__main__":
    train_mpe.make_train_env = make_sequential_train_env
    train_mpe.main(sys.argv[1:])
