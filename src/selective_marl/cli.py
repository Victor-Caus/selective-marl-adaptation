"""Command-line interface for training, evaluation, and reproducible pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_marl.evaluation.benchmark import EvaluationConfig, evaluate
from selective_marl.training import TrainConfig, train

ALGORITHMS = (
    "maddpg",
    "mappo_no_comm",
    "mappo",
    "channel_randomized",
    "rmappo",
    "selective",
)


def _profile(name: str) -> tuple[dict, dict]:
    if name == "smoke":
        return (
            {
                "total_steps": 2_048,
                "rollout_steps": 256,
                "update_epochs": 2,
                "minibatch_size": 128,
                "max_cycles": 15,
                "session_episodes": 8,
                "change_episode": 4,
                "hidden_size": 64,
            },
            {
                "seeds": (101,),
                "sessions_per_condition": 1,
                "session_episodes": 8,
                "change_episode": 4,
                "max_cycles": 15,
            },
        )
    if name == "quick":
        return (
            {
                "total_steps": 20_000,
                "rollout_steps": 1_024,
                "update_epochs": 4,
                "session_episodes": 20,
                "change_episode": 10,
            },
            {
                "seeds": (101, 202, 303),
                "sessions_per_condition": 3,
                "session_episodes": 20,
                "change_episode": 10,
            },
        )
    if name == "research":
        return (
            {
                "total_steps": 500_000,
                "rollout_steps": 2_048,
                "update_epochs": 8,
                "session_episodes": 40,
                "change_episode": 20,
                "hidden_size": 256,
            },
            {
                "seeds": (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010),
                "sessions_per_condition": 10,
                "session_episodes": 40,
                "change_episode": 20,
            },
        )
    raise ValueError(f"unknown profile: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selective-marl", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train one policy")
    train_parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    train_parser.add_argument("--steps", type=int, default=100_000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--output", type=Path, default=Path("results/runs"))

    eval_parser = subparsers.add_parser("evaluate", help="evaluate checkpoints")
    eval_parser.add_argument("checkpoints", nargs="+", type=Path)
    eval_parser.add_argument("--sessions", type=int, default=5)
    eval_parser.add_argument("--output", type=Path, default=Path("results/runs"))
    eval_parser.add_argument("--device", default="auto")

    pipeline_parser = subparsers.add_parser("pipeline", help="train and compare all methods")
    pipeline_parser.add_argument(
        "--profile", choices=("smoke", "quick", "research"), default="quick"
    )
    pipeline_parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    pipeline_parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    pipeline_parser.add_argument("--device", default="auto")
    pipeline_parser.add_argument("--output", type=Path, default=Path("results/runs"))

    subparsers.add_parser("doctor", help="print dependency and accelerator information")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "doctor":
        import mpe2
        import numpy
        import torch

        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "numpy": numpy.__version__,
                    "mpe2": getattr(mpe2, "__version__", "installed"),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None,
                },
                indent=2,
            )
        )
        return
    if args.command == "train":
        path = train(
            TrainConfig(
                algorithm=args.algorithm, seed=args.seed, total_steps=args.steps, device=args.device
            ),
            args.output,
        )
        print(path.resolve())
        return
    if args.command == "evaluate":
        path = evaluate(
            args.checkpoints,
            EvaluationConfig(sessions_per_condition=args.sessions, device=args.device),
            args.output,
        )
        print(path.resolve())
        return

    train_values, evaluation_values = _profile(args.profile)
    checkpoints: list[Path] = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            config = TrainConfig(algorithm=algorithm, seed=seed, device=args.device, **train_values)
            run_path = train(config, args.output)
            checkpoints.append(run_path / "checkpoint.pt")
    evaluation_config = EvaluationConfig(device=args.device, **evaluation_values)
    evaluation_path = evaluate(checkpoints, evaluation_config, args.output)
    print(f"evaluation={evaluation_path.resolve()}")
    print(f"diagnostic_bundle={(evaluation_path / 'diagnostic-bundle.zip').resolve()}")


if __name__ == "__main__":
    main()
