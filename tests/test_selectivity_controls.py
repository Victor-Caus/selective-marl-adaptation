import numpy as np
import pytest
import torch


def adapter(mode):
    pytest.importorskip("onpolicy")
    from test_reference_backend import make_actor

    from selective_marl.adapters_v2 import GuardedAdapters, SemanticMemory

    return GuardedAdapters(make_actor(), SemanticMemory(), mode=mode)


def test_removing_diagnosis_or_reward_changes_only_intended_gate():
    reward = adapter("reward_only_v2")
    selective = adapter("selective_v2")
    confidence = adapter("confidence_only_v2")
    for obj in (reward, selective):
        obj.probabilities = [np.zeros(2)]
        for _ in range(8):
            obj.observe_return([-10, -10])
        for _ in range(3):
            obj.observe_return([-30, -10])
    assert reward.semantic_active.tolist() == [True, False]
    assert not selective.semantic_active.any()
    confidence.probabilities = [np.array([0.9, 0.1])]
    for _ in range(8):
        confidence.observe_return([-10, -10])
    assert confidence.semantic_active.tolist() == [True, False]


def test_no_rollback_retains_activation_after_harm():
    obj = adapter("no_rollback_v2")
    obj.probabilities = [np.ones(2)]
    for _ in range(8):
        obj.observe_return([-10, -10])
    for _ in range(3):
        obj.observe_return([-30, -10])
    for _ in range(4):
        obj.observe_return([-100, -10])
    assert obj.semantic_active.tolist() == [True, False]
    assert obj.rollbacks.tolist() == [0, 0]
    assert obj.trials == [None, None]


def test_shared_stream_uses_receiver_features_for_both_heads():
    obj = adapter("shared_stream_v2")
    obj.semantic_active[:] = True
    obs = np.zeros((2, 21), np.float32)
    obs[:, -10:] = np.eye(10)[[2, 7]]
    seen = []
    hook = obj.actor.rnn.register_forward_hook(lambda m, x, out: seen.append(out[0]))
    with torch.no_grad():
        _, messages, _ = obj.distributions(obs, torch.zeros(2, 1, 64))
        expected = obj.actor.act.action_outs[1](seen[0]).probs
    hook.remove()
    torch.testing.assert_close(messages.probs, expected)


def test_randomization_matches_session_schedule_without_extra_policy_inputs():
    pytest.importorskip("onpolicy")
    from selective_marl.environments.reference_backend import RandomizedReferenceBackendEnv

    a, b = RandomizedReferenceBackendEnv(71), RandomizedReferenceBackendEnv(71)
    try:
        conditions = []
        for episode in range(192):
            np.testing.assert_array_equal(a.reset(), b.reset())
            assert a.env.spec == b.env.spec
            assert a.env.episode_index == episode % 48
            assert 10 <= a.env.spec.change_episode <= 30
            assert a.observation_space[0].shape == (21,)
            if episode % 48 == 0:
                conditions.append(a.env.spec.condition.value)
        assert conditions == ["none", "semantic", "behavioral", "both"]
    finally:
        a.close()
        b.close()
