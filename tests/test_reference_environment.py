import numpy as np

from selective_marl.environments.interventions import InterventionKind
from selective_marl.environments.reference import (
    ACTION_COUNT,
    MESSAGE_COUNT,
    ReferenceSessionEnv,
    decode_action,
    encode_action,
    make_session_spec,
    rotate_movement,
)


def test_action_codec_round_trip() -> None:
    for action in range(ACTION_COUNT):
        movement, message = decode_action(action)
        assert encode_action(movement, message) == action


def test_behavior_transform_preserves_message() -> None:
    for message in range(MESSAGE_COUNT):
        for movement in range(5):
            transformed = encode_action(rotate_movement(movement), message)
            _, transformed_message = decode_action(transformed)
            assert transformed_message == message


def test_semantic_shift_preserves_physical_observation() -> None:
    spec = make_session_spec(InterventionKind.SEMANTIC, change_episode=0, seed=8)
    env = ReferenceSessionEnv(spec)
    try:
        observation = np.arange(21, dtype=np.float32)
        transformed = env._observations({"agent_0": observation})["agent_0"]
        np.testing.assert_array_equal(transformed[:-MESSAGE_COUNT], observation[:-MESSAGE_COUNT])
        assert sorted(transformed[-MESSAGE_COUNT:].tolist()) == sorted(
            observation[-MESSAGE_COUNT:].tolist()
        )
    finally:
        env.close()
