import torch

from selective_marl.maddpg import MADDPGPolicy
from selective_marl.models import build_policy


def test_all_policy_variants_produce_valid_shapes() -> None:
    observations = torch.zeros(2, 21)
    states = torch.zeros(2, 42)
    labels = torch.tensor([0, 1])
    for algorithm in ("mappo", "rmappo", "selective"):
        model = build_policy(algorithm, hidden_size=64)
        hidden = model.initial_hidden(2, torch.device("cpu"))
        output = model.forward_step(observations, states, hidden, labels, oracle_gate=True)
        assert output.actions.shape == (2,)
        assert output.log_probs.shape == (2,)
        assert output.values.shape == (2,)
        if algorithm == "selective":
            assert output.diagnosis_logits is not None
            assert output.diagnosis_logits.shape == (2, 4)


def test_maddpg_policy_produces_two_decentralized_actions() -> None:
    model = MADDPGPolicy(hidden_size=64)
    output = model.forward_step(torch.zeros(2, 21), torch.zeros(2, 42), deterministic=True)
    assert output.actions.shape == (2,)
