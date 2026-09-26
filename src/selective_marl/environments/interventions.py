"""Controlled interventions used by the shift benchmark.

Interventions are represented independently from a learning algorithm. This keeps the
ground-truth cause of a shift explicit and makes it possible to evaluate any baseline under
the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class InterventionKind(StrEnum):
    """Mutually exclusive labels used for diagnosis and evaluation."""

    NONE = "none"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """Ground-truth intervention for one evaluation session."""

    kind: InterventionKind
    change_episode: int
    semantic_permutation: tuple[int, ...] | None = None
    teammate_policy_before: str | None = None
    teammate_policy_after: str | None = None

    def __post_init__(self) -> None:
        if self.change_episode < 0:
            raise ValueError("change_episode must be non-negative")
        semantic_expected = self.kind in {InterventionKind.SEMANTIC, InterventionKind.BOTH}
        behavioral_expected = self.kind in {InterventionKind.BEHAVIORAL, InterventionKind.BOTH}
        if semantic_expected != (self.semantic_permutation is not None):
            raise ValueError("semantic interventions require exactly one semantic permutation")
        has_policy_swap = (
            self.teammate_policy_before is not None and self.teammate_policy_after is not None
        )
        if behavioral_expected != has_policy_swap:
            raise ValueError("behavioral interventions require before and after policy identifiers")
        if has_policy_swap and self.teammate_policy_before == self.teammate_policy_after:
            raise ValueError("behavioral intervention policies must differ")

    def is_active(self, episode: int) -> bool:
        return episode >= self.change_episode


def make_symbol_permutation(size: int, seed: int) -> tuple[int, ...]:
    """Create a deterministic non-identity permutation for a communication channel."""

    if size < 2:
        raise ValueError("a semantic permutation requires at least two symbols")
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    permutation = rng.permutation(size)
    while np.array_equal(identity, permutation):
        permutation = rng.permutation(size)
    return tuple(int(index) for index in permutation)


def apply_symbol_permutation(
    observation: NDArray[np.floating],
    permutation: tuple[int, ...],
    *,
    message_slice: slice,
) -> NDArray[np.floating]:
    """Return an observation with only its received-message dimensions permuted.

    The caller supplies the message slice instead of relying on implicit MPE2 indexes. This
    prevents an environment update from silently changing which features are intervened on.
    """

    result = np.array(observation, copy=True)
    message = result[message_slice]
    if message.ndim != 1:
        raise ValueError("message_slice must select a one-dimensional message vector")
    if len(message) != len(permutation):
        raise ValueError("permutation size must match the selected message dimensions")
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("permutation must contain every message index exactly once")
    result[message_slice] = message[np.asarray(permutation)]
    return result
