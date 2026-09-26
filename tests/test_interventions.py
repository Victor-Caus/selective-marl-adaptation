import numpy as np
import pytest

from selective_marl.environments.interventions import (
    InterventionKind,
    InterventionSpec,
    apply_symbol_permutation,
    make_symbol_permutation,
)


def test_symbol_permutation_changes_only_message_dimensions() -> None:
    observation = np.array([10.0, 20.0, 1.0, 0.0, 0.0])
    changed = apply_symbol_permutation(observation, (1, 2, 0), message_slice=slice(-3, None))

    np.testing.assert_array_equal(changed[:2], observation[:2])
    np.testing.assert_array_equal(changed[-3:], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(observation, np.array([10.0, 20.0, 1.0, 0.0, 0.0]))


def test_generated_permutation_is_deterministic_and_non_identity() -> None:
    first = make_symbol_permutation(10, seed=7)
    second = make_symbol_permutation(10, seed=7)

    assert first == second
    assert first != tuple(range(10))
    assert sorted(first) == list(range(10))


def test_semantic_spec_requires_permutation() -> None:
    with pytest.raises(ValueError, match="semantic interventions"):
        InterventionSpec(kind=InterventionKind.SEMANTIC, change_episode=5)


def test_behavioral_spec_requires_distinct_policies() -> None:
    with pytest.raises(ValueError, match="must differ"):
        InterventionSpec(
            kind=InterventionKind.BEHAVIORAL,
            change_episode=5,
            teammate_policy_before="policy-a",
            teammate_policy_after="policy-a",
        )

