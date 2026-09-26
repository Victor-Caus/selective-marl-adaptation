# Selective Adaptation in Multi-Agent Communication

This project studies a practical failure mode in cooperative multi-agent systems: when
coordination breaks, did the communication protocol change, did the teammate's behavior
change, or did both change?

The first benchmark is built on MPE2 `Simple Reference`. It introduces controlled change
points while preserving the original task dynamics. An agent must detect the change,
attribute its source, and update only the affected component. `Simple World Comm` is the
planned second environment after the complete `Simple Reference` protocol is validated.

## Research question

Can a cooperative agent distinguish semantic shifts from behavioral policy shifts and
recover faster by selectively adapting its communication or teammate-model component?

The benchmark evaluates four conditions:

| Condition | Message semantics | Teammate behavior |
| --- | --- | --- |
| `none` | unchanged | unchanged |
| `semantic` | changed | unchanged |
| `behavioral` | unchanged | changed |
| `both` | changed | changed |

The distinction is deliberately causal: the experiment controller records which mechanism
was intervened on. During evaluation, the learner receives observations, messages, actions,
and rewards, but not the intervention label.

## Current status

- [x] Research protocol and intervention contract
- [x] Core evaluation metrics
- [x] MPE2 `Simple Reference` smoke runner
- [x] Unit tests for interventions and metrics
- [ ] Reproduce random and independent-policy baselines
- [ ] Reproduce MADDPG and MAPPO baselines
- [ ] Add recurrent MAPPO and context baselines
- [ ] Implement semantic/behavioral diagnosis
- [ ] Implement selective adaptation and ablations
- [ ] Validate on `Simple World Comm`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,analysis]"
pytest
selective-marl-smoke --episodes 5 --seed 42
```

The smoke command is not a learning result. It confirms that the installed MPE2 version,
action spaces, observations, seeding, and result recording are working.

## Repository layout

```text
configs/       versioned experiment configurations
docs/          protocol, metrics, literature map, and reproducibility rules
paper/         manuscript material once the protocol is stable
results/       generated summaries and result documentation
src/           benchmark, interventions, evaluation, and experiment code
tests/         deterministic unit and integration tests
```

## Reproducibility policy

Every reported number must be traceable to a committed configuration, code revision,
environment version, seed list, and raw run identifier. Aggregate tables must report the
number of seeds and uncertainty, not only the best run. Failed and incomplete runs are
retained in the experiment manifest and excluded only by documented rules.

See [the experiment protocol](docs/experiment-protocol.md), [metric definitions](docs/metrics.md),
and [the literature map](docs/related-work.md) before adding algorithms or reporting results.

## Citation

This repository is under active development. Release-specific citation metadata will be
archived when the first benchmark version is complete. Until then, use the metadata in
`CITATION.cff` and cite MPE2 and the original MPE/MADDPG work as described in
`docs/related-work.md`.

## License

Code in this repository is released under the MIT License. External environments,
implementations, datasets, and figures retain their original licenses.

