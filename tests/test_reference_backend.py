import numpy as np
import pytest
import torch

pytest.importorskip("onpolicy")

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config

from selective_marl.adapters import OnlineAdapters
from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import ReferenceSessionEnv, make_session_spec
from selective_marl.environments.reference_backend import ReferenceBackendEnv


def test_adapter_translates_movement_sign_and_reward_aggregation():
    adapted = ReferenceBackendEnv(seed=123)
    native = ReferenceSessionEnv(make_session_spec(InterventionKind.NONE, 0, 123), local_ratio=0)
    try:
        initial = adapted.reset()
        obs, _ = native.reset(seed=123, episode_index=0)
        np.testing.assert_allclose(initial, np.stack(list(obs.values())))
        # Historical one-hot index 1 moves right; MPE2 discrete index 2 moves right.
        actions = np.zeros((2, 15))
        actions[:, 1] = 1
        actions[:, 5 + 7] = 1
        actual, rewards, dones, _ = adapted.step(actions)
        expected, native_rewards, _, _, _ = native.step({"agent_0": 37, "agent_1": 37})
        np.testing.assert_allclose(actual, np.stack(list(expected.values())))
        np.testing.assert_allclose(rewards[:, 0], 2 * np.asarray(list(native_rewards.values())))
        assert np.all(actual[:, 0] > 0)
        assert not dones.any()
        for _ in range(24):
            _, _, dones, _ = adapted.step(actions)
        assert dones.all()
    finally:
        adapted.close()
        native.close()


def make_actor():
    torch.set_num_threads(1)
    args = get_config().parse_args([])
    args.use_recurrent_policy = True
    env = ReferenceBackendEnv()
    try:
        return R_MAPPOPolicy(
            args, env.observation_space[0], env.share_observation_space[0], env.action_space[0]
        ).actor
    finally:
        env.close()


@pytest.mark.parametrize("condition", list(InterventionKind))
def test_liam_training_and_evaluation_trajectories_and_team_rewards_match(condition):
    spec = make_session_spec(condition, 0, 731)
    adapted = ReferenceBackendEnv(seed=733)
    adapted.env.spec = spec
    native = ReferenceSessionEnv(spec, local_ratio=0.5)
    rng = np.random.default_rng(737)
    try:
        actual = adapted.reset()
        obs, _ = native.reset(seed=733, episode_index=0)
        np.testing.assert_allclose(actual, np.stack(list(obs.values())))
        for _ in range(25):
            movement, message = rng.integers(0, 5, 2), rng.integers(0, 10, 2)
            actions = np.concatenate((np.eye(5)[movement], np.eye(10)[message]), axis=1)
            encoded = np.asarray([0, 2, 1, 4, 3])[movement] + 5 * message
            actual, rewards, dones, _ = adapted.step(actions)
            obs, native_rewards, terminated, truncated, _ = native.step(
                dict(zip(native.possible_agents, encoded.tolist(), strict=True))
            )
            np.testing.assert_allclose(actual, np.stack(list(obs.values())))
            np.testing.assert_allclose(
                rewards.mean() / 2, np.mean(list(native_rewards.values())), rtol=1e-6
            )
            assert bool(dones.all()) == all(
                terminated[a] or truncated[a] for a in native.possible_agents
            )
    finally:
        adapted.close()
        native.close()


def test_identity_adapters_preserve_original_policy():
    actor = make_actor()
    adapter = OnlineAdapters(actor)
    obs = torch.randn(2, 21)
    hidden = torch.zeros(2, 1, 64)
    with torch.no_grad():
        expected, _, expected_hidden = actor(obs, hidden, torch.ones(2, 1), deterministic=True)
        move, message, actual_hidden = adapter.distributions(obs, hidden)
    actual = torch.stack([move.probs.argmax(-1), message.probs.argmax(-1)], -1)
    torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(expected_hidden, actual_hidden)


def test_selective_update_preserves_frozen_policy_and_inactive_motor():
    actor = make_actor()
    adapter = OnlineAdapters(actor)
    adapter.semantic_active[:] = True
    trajectory = ([], [], [], [])
    initial = torch.zeros(2, 1, 64)
    hidden = initial.clone()
    for i in range(12):
        obs = np.random.randn(2, 21).astype(np.float32)
        with torch.no_grad():
            move, message, hidden = adapter.distributions(obs, hidden)
            action = torch.stack([move.sample(), message.sample()], -1)
            log = move.log_prob(action[:, 0]) + message.log_prob(action[:, 1])
        for buffer, value in zip(
            trajectory, (obs, action.numpy(), log.numpy(), np.array([-i, -i])), strict=True
        ):
            buffer.append(value)
    before = {k: v.clone() for k, v in adapter.actor.state_dict().items()}
    adapter.update(trajectory, initial)
    torch.testing.assert_close(adapter.motor, torch.eye(5).repeat(2, 1, 1))
    assert not torch.equal(adapter.semantic, torch.eye(10).repeat(2, 1, 1))
    torch.testing.assert_close(adapter.semantic.sum(-1), torch.ones(2, 10))
    for name, value in adapter.actor.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_motor_identification_uses_local_transitions_and_preserves_other_agent():
    adapter = OnlineAdapters(make_actor())
    rng = np.random.default_rng(5)
    effects = np.asarray([[0, 0], [0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]])
    remap = np.asarray([0, 3, 4, 2, 1])
    for step in range(160):
        obs = np.zeros((2, 21), dtype=np.float32)
        obs[:, :2] = rng.normal(size=(2, 2))
        action = np.zeros((2, 2), dtype=int)
        action[:, 0] = 1 + step % 4
        executed = action[:, 0].copy()
        if step >= 100:
            executed[1] = remap[executed[1]]
        next_obs = obs.copy()
        next_obs[:, :2] = 0.75 * obs[:, :2] + effects[executed]
        adapter.observe_transition(obs, action, next_obs)
    assert adapter.motor_active.tolist() == [False, True]
    torch.testing.assert_close(adapter.motor[0], torch.eye(5))
    np.testing.assert_array_equal(adapter.motor[1].detach().numpy().argmax(-1), np.argsort(remap))
