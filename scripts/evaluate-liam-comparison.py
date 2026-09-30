"""New paired AHT evaluation: one controlled agent, identical fixed R-MAPPO partner."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from selective_marl.adapters import OnlineAdapters
from selective_marl.adapters_v2 import GuardedAdapters, SemanticMemory
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.environments.reference_backend import HISTORICAL_TO_MPE2
from selective_marl.liam_reference import hidden as liam_hidden
from selective_marl.liam_reference import make_agent


def main():
    p = argparse.ArgumentParser()
    for name in ("actor", "memory", "matched", "long", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--seed", type=int, default=16000001)
    cli = p.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "base_evaluator", Path(__file__).with_name("evaluate-online-adapters.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor, _ = module.actor_from(cli.actor)
    mem = torch.load(cli.memory, map_location="cpu", weights_only=False)
    memory = SemanticMemory(mem["hidden_size"])
    memory.load_state_dict(mem["model_state"])
    memory.eval()
    memory.requires_grad_(False)
    (cli.output / "config.json").write_text(json.dumps(vars(cli), default=str, indent=2))
    modes = [
        "frozen",
        "motor_only",
        "joint_v2",
        "unguarded_v2",
        "selective_v2",
        "liam_matched",
        "liam_long",
    ]
    metrics = []
    with (cli.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mode", "condition", "session", "seed", "episode", "return"]
        )
        writer.writeheader()
        for mode in modes:
            liam = None
            if mode.startswith("liam"):
                liam = make_agent()
                checkpoint = cli.matched if mode == "liam_matched" else cli.long
                liam.load_params(
                    torch.load(checkpoint, map_location="cpu", weights_only=False)["agent_params"]
                )
            for condition in ("none", "semantic", "behavioral", "both"):
                for session in range(cli.sessions):
                    seed = cli.seed + 1000 * session
                    torch.manual_seed(seed)
                    peer_rng = torch.Generator().manual_seed(seed + 70000000)
                    change = int(np.random.default_rng(seed).integers(12, 36))
                    env = ReferenceSessionEnv(
                        make_session_spec(InterventionKind(condition), change, seed)
                    )
                    adapter = (
                        OnlineAdapters(actor, "frozen")
                        if mode == "frozen" or liam
                        else GuardedAdapters(actor, memory, mode, mem["threshold"])
                    )
                    returns, applications = [], 0
                    try:
                        for episode in range(60):
                            observations, _ = env.reset(seed=seed + episode, episode_index=episode)
                            h, peer_h = torch.zeros(2, 1, 64), torch.zeros(1, 1, 64)
                            context, previous = liam_hidden(1), torch.zeros(1, 15)
                            if isinstance(adapter, GuardedAdapters):
                                adapter.begin_episode()
                            total, applied = np.zeros(2), False
                            while env.agents:
                                obs = np.stack([observations[a] for a in env.possible_agents])
                                with torch.no_grad():
                                    peer_features, peer_h = actor.rnn(
                                        actor.base(torch.from_numpy(obs[:1])),
                                        peer_h,
                                        torch.ones(1, 1),
                                    )
                                    peer_action = [
                                        int(
                                            torch.multinomial(
                                                head(peer_features).probs, 1, generator=peer_rng
                                            )
                                        )
                                        for head in actor.act.action_outs
                                    ]
                                    if liam:
                                        action, a1, a2, _, context = liam.act(
                                            torch.from_numpy(obs[1:]), previous, context
                                        )
                                        previous = action[0]
                                        own_action = [int(a1.argmax(-1)), int(a2.argmax(-1))]
                                    else:
                                        move, message, h = adapter.distributions(obs, h)
                                        own_action = [
                                            int(move.sample()[1]),
                                            int(message.sample()[1]),
                                        ]
                                        applied |= bool(
                                            not torch.allclose(adapter.motor[1], torch.eye(5))
                                            or (
                                                isinstance(adapter, GuardedAdapters)
                                                and adapter.memory is not None
                                                and adapter.semantic_active[1]
                                                and obs[1, -10:].sum() > 0
                                            )
                                        )
                                actions = np.asarray([peer_action, own_action])
                                product = HISTORICAL_TO_MPE2[actions[:, 0]] + 5 * actions[:, 1]
                                observations, rewards, _, _, _ = env.step(
                                    dict(zip(env.possible_agents, product.tolist(), strict=True))
                                )
                                reward = np.array([rewards[a] for a in env.possible_agents])
                                next_obs = np.stack([observations[a] for a in env.possible_agents])
                                if isinstance(adapter, GuardedAdapters):
                                    adapter.feedback(obs, actions, reward, next_obs)
                                total += reward
                            if isinstance(adapter, GuardedAdapters):
                                adapter.observe_return(total)
                            value = float(total.mean())
                            returns.append(value)
                            applications += int(applied)
                            writer.writerow(
                                dict(
                                    mode=mode,
                                    condition=condition,
                                    session=session,
                                    seed=seed,
                                    episode=episode,
                                    **{"return": value},
                                )
                            )
                        f.flush()
                        row = dict(
                            mode=mode,
                            condition=condition,
                            session=session,
                            seed=seed,
                            change_episode=change,
                            pre_return=float(np.mean(returns[change - 5 : change])),
                            post_return=float(np.mean(returns[change:])),
                            applied_episodes=applications,
                        )
                        metrics.append(row)
                        print(json.dumps(row), flush=True)
                    finally:
                        env.close()
    (cli.output / "session-metrics.json").write_text(json.dumps(metrics, indent=2))
    (cli.output / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
