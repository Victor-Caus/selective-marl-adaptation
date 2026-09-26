# Metric definitions

Reward alone cannot establish that a message caused coordination or that an agent correctly
identified a shift. Results are divided into task, detection, diagnosis, recovery, and
preservation metrics.

## Task performance

- **Team return:** mean return across cooperative agents per episode.
- **Success rate:** task-specific successful-target completion rate, added once the success
  condition is verified against MPE2 internals.
- **Pre-change return:** mean return over the fixed window before the intervention.

## Disruption and recovery

- **Performance drop:** pre-change return minus the immediate post-change return.
- **Cumulative regret:** sum of positive gaps from the pre-change baseline after the change.
- **Recovery delay:** episodes until performance remains above a fixed fraction of the
  pre-change baseline for a specified consecutive window.
- **Area under the recovery curve:** normalized post-change performance over a fixed horizon.

## Detection and diagnosis

- **Detection delay:** time from true change point to first accepted detection.
- **False-alarm rate:** detections before a change or in `none` sessions.
- **Macro F1 and balanced accuracy:** primary four-class diagnosis metrics.
- **Confusion matrix:** required to expose semantic/behavioral confusion.
- **Calibration error:** whether predicted class probabilities match empirical frequencies.

## Selectivity and preservation

- **Semantic preservation:** communication performance after a behavioral-only shift relative
  to the no-shift control.
- **Behavioral preservation:** motor/task performance after a semantic-only shift relative to
  the no-shift control.
- **Unnecessary-update rate:** fraction of updates applied to a component whose mechanism did
  not change.
- **Interference cost:** loss attributable to adapting the unaffected component.

## Reporting rules

Every table identifies metric direction, unit, evaluation horizon, seeds, aggregation level,
and uncertainty interval. Best-run plots may be diagnostic artifacts but are never primary
evidence.

