import numpy as np
import pytest
import torch

from selective_marl.adapters_v2 import ConservativeDetector, SemanticMemory, memory_features


def test_detector_requires_sustained_evidence_and_ignores_one_outlier():
    detector = ConservativeDetector()
    for _ in range(8):
        assert not detector.observe([-10, -10]).any()
    assert not detector.observe([-100, -10]).any()
    assert not detector.observe([-10, -10]).any()
    for _ in range(3):
        signal = detector.observe([-30, -10])
    assert signal.tolist() == [True, False]


def test_memory_is_causal_and_does_not_mix_agent_histories():
    torch.manual_seed(8)
    memory = SemanticMemory(16).eval()
    x = torch.randn(12, 2, 38)
    changed = x.clone()
    changed[8:] *= -5
    changed[:, 1] += 10
    with torch.no_grad():
        original, _, _ = memory(x)
        other, _, _ = memory(changed)
    torch.testing.assert_close(original[:8, 0], other[:8, 0])
    assert memory_features(np.zeros((2, 21)), np.zeros((2, 2), int), [0, 0], True).shape == (2, 38)


def test_receiver_adaptation_cannot_change_sender_distribution():
    pytest.importorskip("onpolicy")
    from test_reference_backend import make_actor

    from selective_marl.adapters_v2 import GuardedAdapters

    actor, memory = make_actor(), SemanticMemory()
    normal = GuardedAdapters(actor, memory=memory)
    changed = GuardedAdapters(actor, memory=memory)
    changed.semantic_active[:] = True
    h1 = torch.zeros(2, 1, 64)
    h2 = h1.clone()
    for _ in range(10):
        obs = np.random.randn(2, 21).astype(np.float32)
        obs[:, -10:] = np.eye(10)[[2, 7]]
        with torch.no_grad():
            _, send1, h1 = normal.distributions(obs, h1)
            _, send2, h2 = changed.distributions(obs, h2)
        torch.testing.assert_close(send1.probs, send2.probs)


def test_harmful_trial_is_rolled_back_without_labels():
    pytest.importorskip("onpolicy")
    from test_reference_backend import make_actor

    from selective_marl.adapters_v2 import GuardedAdapters

    adapter = GuardedAdapters(make_actor(), memory=SemanticMemory())
    adapter.probabilities = [np.ones(2)]
    for _ in range(8):
        adapter.observe_return([-10, -10])
    for _ in range(3):
        adapter.observe_return([-30, -10])
    assert adapter.semantic_active.tolist() == [True, False]
    for _ in range(4):
        adapter.observe_return([-80, -10])
    assert adapter.semantic_active.tolist() == [False, False]
    assert adapter.rollbacks.tolist() == [1, 0]
    assert adapter.cooldown[0] > 0


def test_motor_only_ignores_supplied_semantic_checkpoint():
    pytest.importorskip("onpolicy")
    from test_reference_backend import make_actor

    from selective_marl.adapters_v2 import GuardedAdapters

    adapter = GuardedAdapters(make_actor(), memory=SemanticMemory(), mode="motor_only")
    assert adapter.memory is None
    adapter.probabilities = [np.ones(2)]
    for i in range(12):
        adapter.observe_return([-10, -10] if i < 8 else [-80, -80])
    assert not adapter.semantic_active.any()
