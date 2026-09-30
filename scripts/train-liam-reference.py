"""Official LIAM update loop on MPE2 with one controlled agent and fixed partner."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from onpolicy.envs.env_wrappers import DummyVecEnv

from selective_marl.environments.reference_backend import RandomizedReferenceBackendEnv
from selective_marl.liam_reference import COMMIT, hidden, make_agent, make_rollout, official


def evaluator():
    spec = importlib.util.spec_from_file_location(
        "original_evaluator", Path(__file__).with_name("evaluate-online-adapters.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=40000000)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--matched-steps", type=int, default=3420800)
    cli = p.parse_args()
    assert cli.steps % (25 * cli.envs) == 0
    cli.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    actor, _ = evaluator().actor_from(cli.actor)
    # Loading the partner must not determine LIAM initialization.
    torch.manual_seed(cli.seed)
    agent = make_agent()
    _, _, Standardize = official()
    standardize = Standardize(shape=1)
    envs = DummyVecEnv(
        [
            lambda rank=i: RandomizedReferenceBackendEnv(20000000 + cli.seed * 200000 + rank * 1000)
            for i in range(cli.envs)
        ]
    )
    state = dict(
        status="running",
        source_commit=COMMIT,
        seed=cli.seed,
        started=time.time(),
        completed_steps=0,
        partner=str(cli.actor),
        controlled_agent=1,
        total_steps=cli.steps,
        processes=cli.envs,
        label="official LIAM core, dimension/environment port; not original benchmark reproduction",
    )
    (cli.output / "config.json").write_text(json.dumps(vars(cli), default=str, indent=2))

    def save_status():
        (cli.output / "status.json").write_text(json.dumps(state, indent=2))

    def checkpoint(name):
        torch.save(
            dict(
                agent_params=agent.get_params(),
                steps=state["completed_steps"],
                source_commit=COMMIT,
                config=vars(cli),
            ),
            cli.output / name,
        )

    save_status()
    observations = envs.reset()
    try:
        for iteration in range(cli.steps // (25 * cli.envs)):
            h = hidden(cli.envs)
            peer_h = torch.zeros(cli.envs, 1, 64)
            a1, a2 = torch.zeros(cli.envs, 5), torch.zeros(cli.envs, 10)
            previous = torch.cat((a1, a2), -1)
            batches, totals = [], np.zeros(cli.envs)
            for chunk in range(5):
                rollout = make_rollout(cli.envs)
                rollout.initialize(a1, a2, h)
                for _step in range(5):
                    own = torch.as_tensor(observations[:, 1], dtype=torch.float32)
                    peer = torch.as_tensor(observations[:, 0], dtype=torch.float32)
                    with torch.no_grad():
                        action, aa1, aa2, value, h = agent.act(own, previous, h)
                        features, peer_h = actor.rnn(
                            actor.base(peer), peer_h, torch.ones(cli.envs, 1)
                        )
                        peer_move = actor.act.action_outs[0](features).sample().squeeze(-1)
                        peer_message = actor.act.action_outs[1](features).sample().squeeze(-1)
                        peer_action = torch.cat(
                            (
                                torch.nn.functional.one_hot(peer_move, 5),
                                torch.nn.functional.one_hot(peer_message, 10),
                            ),
                            -1,
                        ).float()
                    a1, a2, previous = aa1[0], aa2[0], action[0]
                    joint = torch.stack((peer_action, previous), 1).numpy()
                    next_obs, reward, done, _ = envs.step(joint)
                    team_reward = (
                        reward.mean(axis=1) / 2
                    )  # backend scales the native team return by two
                    rollout.insert(
                        own,
                        a1,
                        a2,
                        value[0],
                        torch.from_numpy(team_reward),
                        torch.from_numpy(done[:, 1, None].astype(np.float32)),
                        peer,
                        peer_action,
                    )
                    totals += team_reward[:, 0]
                    observations = next_obs
                with torch.no_grad():
                    last_value = (
                        torch.zeros(cli.envs, 1)
                        if chunk == 4
                        else agent.compute_value(
                            torch.as_tensor(observations[:, 1], dtype=torch.float32), previous, h
                        )[0]
                    )
                rollout.value_preds[-1] = last_value.detach()
                rollout.compute_returns(0.99, 0.95, standardize)
                batches.append(rollout)
            agent.update(batches)
            if not all(
                torch.isfinite(p).all()
                for net in (agent.actor_critic, agent.encoder, agent.decoder)
                for p in net.parameters()
            ):
                raise RuntimeError("Nonfinite LIAM parameters")
            state["completed_steps"] = (iteration + 1) * 25 * cli.envs
            if state["completed_steps"] == cli.matched_steps:
                checkpoint("matched.pt")
            if iteration % 100 == 0 or state["completed_steps"] == cli.steps:
                state.update(
                    mean_train_return=float(totals.mean()),
                    elapsed_seconds=time.time() - state["started"],
                )
                save_status()
                print(json.dumps(state), flush=True)
                with (cli.output / "learning.jsonl").open("a") as f:
                    f.write(json.dumps(state) + "\n")
                checkpoint("latest.pt")
        checkpoint("final.pt")
        state["status"] = "complete"
        save_status()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save_status()
        raise
    finally:
        envs.close()


if __name__ == "__main__":
    main()
