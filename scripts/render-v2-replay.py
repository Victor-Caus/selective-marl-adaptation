"""Replay one archived V2 evaluation session; no training or checkpoint selection."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from selective_marl.adapters_v2 import GuardedAdapters, SemanticMemory
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.environments.reference_backend import HISTORICAL_TO_MPE2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("site/assets"))
    parser.add_argument("--seed", type=int, default=16000001)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "base_evaluator", Path(__file__).with_name("evaluate-online-adapters.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor, _ = module.actor_from(args.actor)
    checkpoint = torch.load(args.memory, map_location="cpu", weights_only=False)
    memory = SemanticMemory(checkpoint["hidden_size"])
    memory.load_state_dict(checkpoint["model_state"])
    adapter = GuardedAdapters(actor, memory, "selective_v2", checkpoint["threshold"])
    torch.manual_seed(args.seed)
    peer_rng = torch.Generator().manual_seed(args.seed + 70000000)
    change = int(np.random.default_rng(args.seed).integers(12, 36))
    # Fixed illustrative frames: baseline, disruption and five episodes later.
    chosen = [change - 1, change, change + 5]
    env = ReferenceSessionEnv(
        make_session_spec(InterventionKind.BEHAVIORAL, change, args.seed),
        render_mode="rgb_array",
    )
    frames, records = [], []
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    try:
        for episode in range(chosen[-1] + 1):
            obs_dict, _ = env.reset(seed=args.seed + episode, episode_index=episode)
            hidden, peer_hidden = torch.zeros(2, 1, 64), torch.zeros(1, 1, 64)
            adapter.begin_episode()
            totals = np.zeros(2)
            step = 0
            episode_frames = []
            while env.agents:
                obs = np.stack([obs_dict[a] for a in env.possible_agents])
                with torch.no_grad():
                    features, peer_hidden = actor.rnn(
                        actor.base(torch.from_numpy(obs[:1])), peer_hidden, torch.ones(1, 1)
                    )
                    peer = [
                        int(torch.multinomial(head(features).probs, 1, generator=peer_rng))
                        for head in actor.act.action_outs
                    ]
                    move, message, hidden = adapter.distributions(obs, hidden)
                    own = [int(move.sample()[1]), int(message.sample()[1])]
                actions = np.asarray([peer, own])
                product = HISTORICAL_TO_MPE2[actions[:, 0]] + 5 * actions[:, 1]
                obs_dict, rewards, _, _, _ = env.step(
                    dict(zip(env.possible_agents, product.tolist(), strict=True))
                )
                reward = np.asarray([rewards[a] for a in env.possible_agents])
                next_obs = np.stack([obs_dict[a] for a in env.possible_agents])
                adapter.feedback(obs, actions, reward, next_obs)
                totals += reward
                step += 1
                if episode in chosen:
                    scene = Image.fromarray(env.render()).convert("RGB")
                    scene.thumbnail((768, 570))
                    canvas = Image.new("RGB", (800, 768), "#f6f7f9")
                    canvas.paste(scene, ((800 - scene.width) // 2, 95))
                    draw = ImageDraw.Draw(canvas)
                    phase = (
                        "Before change"
                        if episode < change
                        else "Rotation begins"
                        if episode == change
                        else "After adaptation"
                    )
                    draw.text((22, 15), f"V2 | {phase}", font=font, fill="#192533")
                    draw.text(
                        (22, 51),
                        f"MPE2 Simple Reference | seed 1 | episode {episode} | step {step}/25",
                        font=small,
                        fill="#546477",
                    )
                    draw.text(
                        (22, 690),
                        "Agent 0: frozen partner | Agent 1: V2 | movement rotation only",
                        font=small,
                        fill="#192533",
                    )
                    draw.text(
                        (22, 733),
                        "Qualitative replay; aggregate results and limitations are in the paper.",
                        font=small,
                        fill="#546477",
                    )
                    episode_frames.append(np.asarray(canvas))
            adapter.observe_return(totals)
            records.append({"episode": episode, "return": float(totals.mean())})
            if episode_frames:
                frames.extend(episode_frames)
                frames.extend([episode_frames[-1]] * 8)
    finally:
        env.close()
    with args.archive.open(newline="") as handle:
        archived = {
            int(row["episode"]): float(row["return"])
            for row in csv.DictReader(handle)
            if row["mode"] == "selective_v2"
            and row["condition"] == "behavioral"
            and int(row["seed"]) == args.seed
        }
    differences = [abs(row["return"] - archived[row["episode"]]) for row in records]
    args.output.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output / "v2-replay.gif", frames, duration=140, loop=0)
    imageio.mimsave(
        args.output / "v2-replay.mp4",
        frames,
        fps=7,
        codec="libx264",
        macro_block_size=16,
        quality=8,
    )
    Image.fromarray(frames[-1]).save(args.output / "v2-replay.png")
    metadata = {
        "method": "selective_v2",
        "condition": "behavioral",
        "training_seed": 1,
        "scenario_seed": args.seed,
        "change_episode": change,
        "shown_episodes": chosen,
        "selection": (
            "First archived seed/session; episodes tau-1, tau, tau+5. No search for best return."
        ),
        "note": (
            "Illustrative inference replay. No training. "
            "Continuous reward; no binary success label."
        ),
        "max_absolute_return_difference_from_archive": max(differences),
        "displayed_returns": [row for row in records if row["episode"] in chosen],
        "checkpoint_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.actor, args.memory)
        },
        "environment": (
            "Farama MPE2 Simple Reference; original MPE by Lowe et al. / Mordatch & Abbeel"
        ),
    }
    (args.output / "replay.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
