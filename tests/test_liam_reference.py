import pytest
import torch

from selective_marl.liam_reference import SOURCE, hidden, make_agent, make_rollout, official

pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="Optional pinned LIAM source missing")


def test_configurable_dimensions_preserve_exact_official_initialization_and_actions():
    A2C, _, _ = official()
    torch.manual_seed(73)
    original = A2C(18, 128, 20, 5, 18, 5, 3e-4, 7e-4, 0.01, 0.5, 0.5)
    torch.manual_seed(73)
    bridge = make_agent(obs=18, communication=5)
    obs, prev = torch.randn(3, 18), torch.zeros(3, 10)
    torch.manual_seed(12)
    expected = original.act(obs, prev, hidden(3))
    torch.manual_seed(12)
    actual = bridge.act(obs, prev, hidden(3))
    for a, b in zip(actual[:4], expected[:4], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_port_updates_with_five_plus_ten_actions_and_no_actor_gradient_into_encoder():
    agent = make_agent()
    obs = torch.randn(5, 3, 21)
    previous = torch.zeros(5, 3, 15)
    movement = torch.nn.functional.one_hot(torch.zeros(5, 3, dtype=torch.long), 5).float()
    message = torch.nn.functional.one_hot(torch.ones(5, 3, dtype=torch.long), 10).float()
    output = agent.evaluate(
        obs,
        previous,
        movement,
        message,
        hidden(3),
        torch.randn(5, 3, 21),
        torch.cat((movement, message), -1),
    )
    (output[0].sum() + output[1].sum() + output[3].sum()).backward()
    assert all(p.grad is None for p in agent.encoder.parameters())
    (output[4] + output[5]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.encoder.parameters())
    rollout = make_rollout(3)
    assert rollout.actions2.shape == (6, 3, 10)
    assert rollout.modelled_acts.shape == (5, 3, 15)


def test_bridge_preserves_official_optimizer_update_at_matching_dimensions():
    import copy

    torch.set_num_threads(1)
    A2C, _, _ = official()
    torch.manual_seed(91)
    original = A2C(21, 128, 20, 5, 21, 5, 3e-4, 7e-4, 0.01, 0.5, 0.5)
    torch.manual_seed(91)
    bridge = make_agent(communication=5)
    rollout = make_rollout(3, communication=5)
    rollout.obs.normal_()
    rollout.modelled_obs.normal_()
    rollout.returns.normal_()
    rollout.value_preds.normal_()
    for actions in (rollout.actions1, rollout.actions2):
        actions.copy_(torch.nn.functional.one_hot(torch.randint(5, (6, 3)), 5))
    rollout.modelled_acts.copy_(torch.cat((rollout.actions1[1:], rollout.actions2[1:]), -1))
    before = copy.deepcopy(bridge.get_params())
    original.update([copy.deepcopy(rollout)])
    bridge.update([copy.deepcopy(rollout)])
    for network, values in bridge.get_params().items():
        assert any(not torch.equal(value, before[network][key]) for key, value in values.items())
        for key, value in values.items():
            torch.testing.assert_close(value, original.get_params()[network][key], rtol=0, atol=0)
