"""Live playback of trained policies in controlled Simple Reference sessions."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.evaluation.benchmark import load_policy
from selective_marl.training import choose_device, set_seed


def watch_checkpoint(
    checkpoint: Path,
    *,
    condition: InterventionKind = InterventionKind.SEMANTIC,
    episodes: int = 2,
    change_episode: int = 1,
    max_cycles: int = 25,
    seed: int = 101,
    fps: float = 12.0,
    deterministic: bool = True,
    oracle_gate: bool = False,
    device_name: str = "auto",
) -> None:
    """Render a checkpoint before and after a controlled intervention."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 <= change_episode <= episodes:
        raise ValueError("change_episode must be between zero and episodes")
    if fps <= 0:
        raise ValueError("fps must be positive")

    device = choose_device(device_name)
    set_seed(seed)
    policy = load_policy(checkpoint, device, oracle_gate=oracle_gate)
    if policy.model is None:
        raise ValueError("watch requires a trained checkpoint")

    spec = make_session_spec(condition, change_episode, seed)
    env = ReferenceSessionEnv(
        spec,
        max_cycles=max_cycles,
        render_mode="human",
        mask_messages=policy.mask_messages,
    )
    hidden = policy.model.initial_hidden(len(env.possible_agents), device)
    frame_delay = 1.0 / fps
    try:
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode, episode_index=episode)
            totals = {agent: 0.0 for agent in env.possible_agents}
            predictions: list[int] = []
            env.render()
            while env.agents:
                agents = env.possible_agents
                obs = np.stack([observations[agent] for agent in agents])
                state = np.repeat(env.global_state(observations)[None, :], len(agents), axis=0)
                labels = torch.full(
                    (len(agents),), env.true_label(), dtype=torch.long, device=device
                )
                with torch.no_grad():
                    output = policy.model.forward_step(
                        torch.as_tensor(obs, dtype=torch.float32, device=device),
                        torch.as_tensor(state, dtype=torch.float32, device=device),
                        hidden,
                        labels,
                        deterministic=deterministic,
                        oracle_gate=oracle_gate,
                    )
                hidden = output.hidden
                actions = {
                    agent: int(output.actions[index]) for index, agent in enumerate(agents)
                }
                if output.diagnosis_logits is not None:
                    predictions.append(int(output.diagnosis_logits.argmax(-1)[0]))
                observations, rewards, _, _, _ = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                env.render()
                time.sleep(frame_delay)

            mean_return = float(np.mean(list(totals.values())))
            phase = "post-change" if episode >= change_episode else "pre-change"
            diagnosis = (
                str(max(set(predictions), key=predictions.count))
                if predictions
                else "not available"
            )
            print(
                f"episode={episode} phase={phase} true_label={env.true_label()} "
                f"diagnosis={diagnosis} team_return={mean_return:.3f}"
            )
    finally:
        env.close()
