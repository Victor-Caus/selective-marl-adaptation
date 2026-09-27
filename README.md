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

- [x] Research protocol and mechanism-isolated interventions
- [x] Structured metrics and diagnostic bundles
- [x] Random and no-communication controls
- [x] Discrete MADDPG and MAPPO baselines
- [x] Channel-randomized MAPPO and recurrent MAPPO/GRU
- [x] Factorized semantic/behavioral contexts with four-class diagnosis
- [x] Selective context updates and oracle-gated ablation
- [x] Automated plots, CSV tables, GIFs, checkpoints, and CI
- [x] Pinned official MAPPO source and isolated Windows reproduction runner
- [ ] Run the preregistered long experiments on multiple seeds
- [ ] Review evidence before opening the `Simple World Comm` stage

The in-house learning algorithms are preliminary reimplementations. Reference reproduction
and parity checks now precede any scientific comparison; see
[the reproduction plan](docs/reproduction-plan.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,analysis]"
pytest
selective-marl-smoke --episodes 5 --seed 42
```

On Windows, the complete engineering check is:

```powershell
.\scripts\setup.ps1
.\scripts\run-smoke.ps1
```

The smoke profile trains every method briefly and verifies every artifact. It is not a
scientific result. For an exploratory run and the preregistered long run:

```powershell
.\scripts\run-quick.ps1
.\scripts\run-research.ps1
```

See [the runbook](docs/runbook.md) for expected duration, output files, and how to share a
diagnostic bundle.

Before using the preliminary implementations for comparisons, reproduce the official
R-MAPPO baseline:

```powershell
.\scripts\setup-reference-mappo.ps1
.\scripts\run-reference-mappo.ps1 -Profile smoke
.\scripts\run-reference-mappo.ps1 -Profile full
```

The local Windows runner keeps all 128 official environments but advances them sequentially
because spawning 32 or 128 Python processes exhausts this host's virtual memory. See
[the recorded host deviations](reproductions/host-deviations.md).

## Watch a trained policy

Open a live MPE2 window using the newest checkpoint for an algorithm:

```powershell
.\scripts\watch-latest.ps1 -Algorithm mappo -Condition semantic
.\scripts\watch-latest.ps1 -Algorithm selective -Condition behavioral
```

The default playback shows one episode before and one episode after the selected controlled
shift. Existing pipeline GIFs remain available under each evaluation directory's `videos/`.

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

## Generated artifacts

Every pipeline run creates an ignored directory under `results/runs/` containing:

- `metadata.json`, resolved configuration, and console log;
- episode-level JSONL and CSV records;
- model checkpoint for each trained policy;
- aggregate `summary.csv`, `summary.json`, and `REPORT.md`;
- recovery curves and comparison plots;
- rendered GIFs for qualitative inspection;
- `diagnostic-bundle.zip` containing all shareable diagnostics except large checkpoints.

`.venv`, `.pytest_cache`, `.ruff_cache`, `.cache`, raw runs, checkpoints, and generated GIFs
are ignored by Git. They can appear in an editor's file tree without being committed.

## Citation

This repository is under active development. Release-specific citation metadata will be
archived when the first benchmark version is complete. Until then, use the metadata in
`CITATION.cff` and cite MPE2 and the original MPE/MADDPG work as described in
`docs/related-work.md`.

## License

Code in this repository is released under the MIT License. External environments,
implementations, datasets, and figures retain their original licenses.

