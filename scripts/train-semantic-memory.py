"""Train semantic context from training-only interventions, then calibrate on none.

Clean received symbols and semantic labels are supervised targets, never inputs.
Each agent receives only its own observation, previous action/reward and reset bit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from selective_marl.adapters_v2 import (
    ConservativeDetector,
    GuardedAdapters,
    SemanticMemory,
    memory_features,
)
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.environments.reference_backend import HISTORICAL_TO_MPE2


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "adapter_evaluator", Path(__file__).with_name("evaluate-online-adapters.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(actor, args, count, seed, episodes, normal_only=False):
    features, targets, labels, resets, episode_returns, session_info = [], [], [], [], [], []
    for session in range(count):
        session_seed = seed + session * 101
        rng = np.random.default_rng(session_seed)
        condition = (
            "none" if normal_only else ("none", "semantic", "behavioral", "both")[session % 4]
        )
        change = int(rng.integers(10, min(31, episodes - 5)))
        spec = make_session_spec(InterventionKind(condition), change, session_seed)
        env = ReferenceSessionEnv(spec)
        policy = GuardedAdapters(actor, mode="motor_only")
        torch.manual_seed(session_seed)
        xs, ys, ls, rs, returns = [], [], [], [], []
        try:
            for episode in range(episodes):
                obs_dict, _ = env.reset(seed=session_seed + episode, episode_index=episode)
                policy.begin_episode()
                hidden = torch.zeros(2, args.recurrent_N, args.hidden_size)
                total = np.zeros(2)
                while env.agents:
                    obs = np.stack([obs_dict[a] for a in env.possible_agents])
                    xs.append(
                        memory_features(
                            obs,
                            policy.previous_actions,
                            policy.previous_rewards,
                            policy.episode_reset,
                        )
                    )
                    active = episode >= change and condition in {"semantic", "both"}
                    clean = obs[:, -10:]
                    if active:
                        clean = clean[:, np.argsort(spec.semantic_permutation)]
                    ys.append(np.where(clean.sum(-1) > 0, clean.argmax(-1), -100))
                    ls.append(np.full(2, float(active)))
                    rs.append(np.full(2, len(xs) == 1))
                    with torch.no_grad():
                        move, message, hidden = policy.distributions(obs, hidden)
                        action = torch.stack([move.sample(), message.sample()], -1).numpy()
                    product = HISTORICAL_TO_MPE2[action[:, 0]] + 5 * action[:, 1]
                    obs_dict, reward_dict, _, _, _ = env.step(
                        dict(zip(env.possible_agents, product.tolist(), strict=True))
                    )
                    next_obs = np.stack([obs_dict[a] for a in env.possible_agents])
                    rewards = np.asarray([reward_dict[a] for a in env.possible_agents])
                    policy.feedback(obs, action, rewards, next_obs)
                    total += rewards
                returns.append(total)
                policy.observe_return(total)
        finally:
            env.close()
        features.append(xs)
        targets.append(ys)
        labels.append(ls)
        resets.append(rs)
        episode_returns.append(returns)
        session_info.append(
            dict(
                seed=session_seed,
                condition=condition,
                change_episode=change,
                permutation=spec.semantic_permutation,
            )
        )
        if session % 8 == 0:
            print(f"collect session={session + 1}/{count} condition={condition}", flush=True)
    return dict(
        x=np.asarray(features, np.float32),
        y=np.asarray(targets, np.int64),
        label=np.asarray(labels, np.float32),
        returns=np.asarray(episode_returns, np.float32),
        info=session_info,
    )


def tensorize(data, device):
    # Batch contains independent agent histories; no teammate observations are concatenated.
    return [
        torch.as_tensor(data[k], device=device).transpose(0, 1).flatten(1, 2)
        for k in ("x", "y", "label")
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sessions", type=int, default=256)
    p.add_argument("--validation-sessions", type=int, default=32)
    p.add_argument("--calibration-sessions", type=int, default=64)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    cli = p.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    config = {
        **vars(cli),
        "actor": str(cli.actor),
        "output": str(cli.output),
        "supervision": "training-only clean received symbol and semantic-change label",
        "runtime_inputs": "own obs, previous own action/reward, episode reset bit",
        "base_actor_frozen": True,
    }
    (cli.output / "config.json").write_text(json.dumps(config, indent=2))
    state = dict(status="collecting", started=time.time())

    def status(**kwargs):
        state.update(kwargs)
        (cli.output / "status.json").write_text(json.dumps(state, indent=2))

    status()
    actor, args = load_evaluator().actor_from(cli.actor)
    try:
        train = collect(actor, args, cli.sessions, 1000000 + cli.seed * 100000, cli.episodes)
        validation = collect(
            actor, args, cli.validation_sessions, 3000000 + cli.seed * 100000, cli.episodes
        )
        calibration = collect(
            actor, args, cli.calibration_sessions, 5000000 + cli.seed * 100000, cli.episodes, True
        )
        for name, data in (
            ("train", train),
            ("validation", validation),
            ("calibration", calibration),
        ):
            np.savez_compressed(
                cli.output / f"{name}.npz", **{k: v for k, v in data.items() if k != "info"}
            )
            (cli.output / f"{name}-sessions.json").write_text(json.dumps(data["info"], indent=2))
        # Threshold chosen exclusively from normal calibration trajectories. No test labels used.
        maxima = []
        for returns in calibration["returns"]:
            detector = ConservativeDetector()
            maximum = 0.0
            for value in returns:
                detector.observe(value)
                eligible = detector.cusum[detector.bad_streak >= 3]
                maximum = max(maximum, float(eligible.max()) if len(eligible) else 0.0)
            maxima.append(maximum)
        threshold = max(6.0, float(np.quantile(maxima, 0.99, method="higher")) + 0.1)
        (cli.output / "calibration.json").write_text(
            json.dumps(
                dict(
                    threshold=threshold,
                    confidence=0.8,
                    maxima=maxima,
                    caveat="empirical normal-calibration bound; not a statistical guarantee",
                ),
                indent=2,
            )
        )
        device = torch.device(cli.device)
        model = SemanticMemory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        tx, ty, tl = tensorize(train, device)
        vx, vy, vl = tensorize(validation, device)
        best = float("inf")
        status(status="training", threshold=threshold)
        for epoch in range(cli.epochs):
            model.train()
            losses = []
            # Shuffle whole agent histories, never individual recurrent transitions.
            order = torch.randperm(tx.shape[1], device=device)
            for start in range(0, len(order), 32):
                batch = order[start : start + 32]
                hidden = None
                for t in range(0, len(tx), 100):
                    x, y, label = (
                        tx[t : t + 100, batch],
                        ty[t : t + 100, batch],
                        tl[t : t + 100, batch],
                    )
                    decoded, diagnosis, hidden = model(x, hidden)
                    ce = F.cross_entropy(
                        decoded.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none"
                    ).reshape_as(y)
                    weights = (1 + 2 * label) * (y != -100)
                    loss = (ce * weights).sum() / weights.sum().clamp_min(1)
                    loss += F.binary_cross_entropy_with_logits(diagnosis, label)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    hidden = hidden.detach()
                    losses.append(float(loss.detach()))
            model.eval()
            total, correct, count = 0.0, 0, 0
            with torch.no_grad():
                hidden = None
                for t in range(0, len(vx), 100):
                    decoded, diagnosis, hidden = model(vx[t : t + 100], hidden)
                    y, label = vy[t : t + 100], vl[t : t + 100]
                    total += float(
                        F.cross_entropy(decoded.flatten(0, 1), y.flatten(), ignore_index=-100)
                        + F.binary_cross_entropy_with_logits(diagnosis, label)
                    )
                    mask = (label > 0) & (y != -100)
                    correct += int(((decoded.argmax(-1) == y) & mask).sum())
                    count += int(mask.sum())
            row = dict(
                epoch=epoch + 1,
                train_loss=float(np.mean(losses)),
                validation_loss=total,
                shifted_symbol_accuracy=correct / max(count, 1),
            )
            print(json.dumps(row), flush=True)
            with (cli.output / "epochs.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if total < best:
                best = total
                torch.save(
                    dict(
                        model_state=model.state_dict(),
                        hidden_size=128,
                        threshold=threshold,
                        epoch=epoch + 1,
                        config=config,
                    ),
                    cli.output / "memory.pt",
                )
        status(
            status="complete",
            elapsed_seconds=time.time() - state["started"],
            best_validation_loss=best,
        )
    except BaseException as exc:
        status(status="failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
