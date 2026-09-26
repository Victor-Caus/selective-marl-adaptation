import pytest

from selective_marl.evaluation.metrics import (
    classification_accuracy,
    compute_shift_metrics,
    detection_delay,
)


def test_shift_metrics_capture_drop_and_recovery() -> None:
    metrics = compute_shift_metrics(
        [10, 10, 10, 4, 5, 8, 10, 10, 10],
        change_episode=3,
        pre_window=3,
        post_window=2,
        recovery_threshold=0.9,
        recovery_sustain=2,
    )

    assert metrics.pre_change_return == pytest.approx(10.0)
    assert metrics.immediate_post_change_return == pytest.approx(4.5)
    assert metrics.performance_drop == pytest.approx(5.5)
    assert metrics.cumulative_regret == pytest.approx(13.0)
    assert metrics.recovery_episode == 6
    assert metrics.recovery_delay == 3


def test_shift_metrics_report_missing_recovery() -> None:
    metrics = compute_shift_metrics([10, 10, 4, 5, 6], change_episode=2, recovery_sustain=2)
    assert metrics.recovery_episode is None
    assert metrics.recovery_delay is None


def test_diagnosis_metrics() -> None:
    assert classification_accuracy(["none", "semantic"], ["none", "both"]) == 0.5
    assert detection_delay(10, 14) == 4
    assert detection_delay(10, None) is None
