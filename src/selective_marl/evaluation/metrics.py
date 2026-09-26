"""Metrics that separate task performance, diagnosis, and adaptation speed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ShiftMetrics:
    pre_change_return: float
    immediate_post_change_return: float
    performance_drop: float
    cumulative_regret: float
    recovery_episode: int | None
    recovery_delay: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def compute_shift_metrics(
    episode_returns: ArrayLike,
    *,
    change_episode: int,
    pre_window: int = 5,
    post_window: int = 5,
    recovery_threshold: float = 0.95,
    recovery_sustain: int = 3,
) -> ShiftMetrics:
    """Summarize disruption and recovery around a known evaluation change point."""

    returns = np.asarray(episode_returns, dtype=float)
    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("episode_returns must be a non-empty one-dimensional sequence")
    if not 0 < change_episode < len(returns):
        raise ValueError("change_episode must have observations before and after it")
    if min(pre_window, post_window, recovery_sustain) < 1:
        raise ValueError("all metric windows must be positive")
    if not 0 < recovery_threshold <= 1:
        raise ValueError("recovery_threshold must be in (0, 1]")

    pre = returns[max(0, change_episode - pre_window) : change_episode]
    post = returns[change_episode : min(len(returns), change_episode + post_window)]
    baseline = float(np.mean(pre))
    immediate = float(np.mean(post))
    performance_drop = baseline - immediate
    cumulative_regret = float(np.maximum(0.0, baseline - returns[change_episode:]).sum())

    target = baseline * recovery_threshold
    recovery_episode: int | None = None
    for index in range(change_episode, len(returns) - recovery_sustain + 1):
        window = returns[index : index + recovery_sustain]
        if bool(np.all(window >= target)):
            recovery_episode = index
            break

    recovery_delay = None if recovery_episode is None else recovery_episode - change_episode
    return ShiftMetrics(
        pre_change_return=baseline,
        immediate_post_change_return=immediate,
        performance_drop=performance_drop,
        cumulative_regret=cumulative_regret,
        recovery_episode=recovery_episode,
        recovery_delay=recovery_delay,
    )


def classification_accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    """Compute intervention-classification accuracy with strict length checking."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must have the same length")
    if not expected:
        raise ValueError("at least one label is required")
    return float(np.mean(np.asarray(expected) == np.asarray(predicted)))


def detection_delay(change_episode: int, detected_episode: int | None) -> int | None:
    """Return non-negative detection delay, or None when no change was detected."""

    if detected_episode is None:
        return None
    return max(0, detected_episode - change_episode)
