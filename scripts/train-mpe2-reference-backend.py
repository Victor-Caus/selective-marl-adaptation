"""Train our MPE2 control with the unmodified pinned R-MAPPO optimizer."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from onpolicy.runner.shared.mpe_runner import MPERunner

from selective_marl.environments.reference_backend import (
    RandomizedReferenceBackendEnv,
    ReferenceBackendEnv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-actor", type=Path)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    cli = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / ".external/reference-sources/marlbenchmark-on-policy"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "de66d7a4b23fac2513f56f96f73b3f5cb96695ac":
        raise RuntimeError(f"Unexpected reference commit: {commit}")
    cli.output.mkdir(parents=True, exist_ok=False)
    args = get_config().parse_args([])
    overrides = dict(
        env_name="MPE",
        algorithm_name="rmappo",
        experiment_name="mpe2-reference-backend",
        scenario_name="simple_reference",
        num_agents=2,
        num_landmarks=3,
        seed=cli.seed,
        n_training_threads=1,
        n_rollout_threads=cli.envs,
        num_mini_batch=1,
        episode_length=25,
        num_env_steps=cli.steps,
        ppo_epoch=15,
        gain=0.01,
        lr=cli.learning_rate,
        critic_lr=cli.learning_rate,
        use_wandb=False,
        use_recurrent_policy=True,
        use_naive_recurrent_policy=False,
        log_interval=10,
        save_interval=100,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    (cli.output / "config.json").write_text(json.dumps(vars(args), indent=2))
    metadata = dict(
        status="running",
        reference_commit=commit,
        backend="pinned official R-MAPPO on MPE2",
        reward="sum of agent distance rewards",
        requested_environment_steps=cli.steps,
        actual_environment_steps=cli.steps // (25 * cli.envs) * (25 * cli.envs),
        training_seed=cli.seed,
        training_conditions=["none", "semantic", "behavioral", "both"]
        if cli.randomize
        else ["none"],
        initial_actor=str(cli.initial_actor) if cli.initial_actor else None,
        initialization="actor warm start; fresh critic and ValueNorm"
        if cli.initial_actor
        else "scratch",
    )
    (cli.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    env_class = RandomizedReferenceBackendEnv if cli.randomize else ReferenceBackendEnv
    factories = [lambda rank=i: env_class(cli.seed + 1000 * rank) for i in range(cli.envs)]
    envs = (SubprocVecEnv if cli.envs > 1 else DummyVecEnv)(factories)
    started = time.time()
    runner = None
    try:
        runner = MPERunner(
            dict(
                all_args=args,
                envs=envs,
                eval_envs=None,
                num_agents=2,
                device=torch.device(cli.device),
                run_dir=cli.output,
            )
        )
        if cli.initial_actor:
            runner.policy.actor.load_state_dict(
                torch.load(cli.initial_actor, map_location=cli.device, weights_only=True)
            )
        runner.run()
        runner.writter.export_scalars_to_json(str(cli.output / "learning-curves.json"))
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        envs.close()
        if runner is not None:
            runner.writter.close()
        metadata["elapsed_seconds"] = time.time() - started
        (cli.output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
