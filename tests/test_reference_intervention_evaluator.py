import numpy as np

from selective_marl.reference_protocol import (
    permute_received_messages,
    rotate_teammate_movement,
)


def test_semantic_shift_changes_only_received_message_dimensions() -> None:
    observations = np.arange(42, dtype=np.float32).reshape(2, 21)
    permutation = np.asarray((1, 2, 3, 4, 5, 6, 7, 8, 9, 0))

    changed = permute_received_messages(observations, permutation)

    np.testing.assert_array_equal(changed[:, :11], observations[:, :11])
    np.testing.assert_array_equal(changed[:, -10:], observations[:, -10:][:, permutation])


def test_behavior_shift_changes_only_teammate_movement() -> None:
    actions = np.asarray(((1, 7), (2, 4)), dtype=np.int64)

    changed = rotate_teammate_movement(actions)

    np.testing.assert_array_equal(changed[0], actions[0])
    assert changed[1, 0] == 4
    assert changed[1, 1] == actions[1, 1]
