# Training and diagnostics runbook

## One-time setup

From the repository root on Windows:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m selective_marl doctor
.\.venv\Scripts\python.exe -m pytest
```

`doctor` reports the exact PyTorch, NumPy, MPE2, and accelerator configuration. The current
implementation runs on CPU and automatically uses CUDA when a compatible PyTorch build and
GPU are available.

## Official MAPPO reference

Set up the separately pinned legacy-compatible environment once, then run the integrity,
pilot, and full profiles in order:

```powershell
.\scripts\setup-reference-mappo.ps1
.\scripts\run-reference-mappo.ps1 -Profile smoke
.\scripts\run-reference-mappo.ps1 -Profile pilot
.\scripts\run-reference-mappo.ps1 -Profile full
```

These commands execute the pinned `marlbenchmark/on-policy` source, not the preliminary
in-house PPO. They save the exact command, dependency freeze, console log, official
checkpoints, TensorBoard export, normalized `metrics.csv`, `analysis.json`, `REPORT.md`, and
`learning-curve.png` under ignored `results/reference-runs/`.

After training, evaluate a copied official checkpoint against a random policy and a causal
communication-silencing ablation:

```powershell
.\.external\tools\micromamba\micromamba.exe run `
  -p .\.external\envs\mappo-reference python `
  .\scripts\evaluate-reference-mappo.py `
  --model-dir <path-to-models> --output-dir <path-to-evaluation> `
  --episodes 200 --save-gif
```

This evaluator is project code that loads the pinned official actor and environment; it is
not copied from the reference repository. It reports raw episode returns, means, standard
deviations, 95% confidence-interval half-widths, message usage, and the trained-minus-silenced
return difference.

The full local Windows profile keeps the official 3,000,000-step budget, 128 environments,
3,200-step batches, and update count. Tests with 128 and 32 spawned workers exhausted host
virtual memory, so a small project adapter advances the 128 official environments
sequentially through the reference `DummyVecEnv`. See `reproductions/host-deviations.md`
before interpreting parity.

## Profiles

### Smoke

```powershell
.\scripts\run-smoke.ps1
```

Trains each algorithm for only 2,048 agent steps and evaluates one session per condition.
Use this after installation or code changes. Never cite its scores.

### Quick

```powershell
.\scripts\run-quick.ps1
```

Uses 20,000 steps per method and three evaluation sessions per condition. It is suitable for
debugging learning curves and selecting reasonable hyperparameters, but not for final claims.

### Research

```powershell
.\scripts\run-research.ps1
```

Uses 500,000 steps per method, three independent training seeds, and ten evaluation sessions
per condition. CPU execution may take many hours. Let the process finish; each algorithm
writes its own checkpoint and logs before the next begins.

## Algorithms

- `random`: evaluation-only lower bound;
- `maddpg`: discrete actor-critic with decentralized actors and centralized critics;
- `mappo_no_comm`: communication ablation;
- `mappo`: feed-forward shared actor and centralized critic;
- `channel_randomized`: MAPPO trained under per-session symbol permutations;
- `rmappo`: MAPPO with persistent GRU context;
- `selective`: separate semantic and behavioral recurrent contexts plus diagnosis;
- `selective-oracle`: evaluation ablation using the true intervention class for routing.

## Outputs

Runs are written below `results/runs/<UTC timestamp>-<name>/`.

Training directories contain `checkpoint.pt`, `episodes.jsonl`, `train_metrics.jsonl`,
`train_summary.json`, `metadata.json`, and the resolved configuration. Evaluation directories
contain raw `episodes.csv`, per-session metrics, confusion matrices, plots, GIFs, `REPORT.md`,
and `diagnostic-bundle.zip`.

To locate the newest bundle:

```powershell
.\scripts\latest-diagnostics.ps1
```

The ZIP intentionally excludes checkpoints. It contains logs, metadata, configurations,
tables, plots, videos, and the generated report and is the preferred artifact to share for
analysis.

## Interpreting results

The primary comparison is not final reward alone. Examine performance drop, cumulative
regret, recovery delay, recovery rate, four-class confusion matrices, and false alarms in the
no-change condition. Compare `selective` with recurrent MAPPO, channel randomization, and
`selective-oracle`. A result is not evidence until it is stable across seeds and accompanied
by uncertainty estimates.

## Live visualization

Watch the newest trained checkpoint for an algorithm:

```powershell
.\scripts\watch-latest.ps1 -Algorithm mappo -Condition semantic
```

The viewer preserves recurrent state between episodes, matching an evaluation session. By
default episode 0 is pre-change and episode 1 is post-change. Use `-OracleGate` only for the
explicit oracle ablation of a selective checkpoint.

Previously generated GIFs can be opened directly, for example:

```powershell
Invoke-Item .\results\runs\20260927T085256Z-evaluation\videos\mappo-semantic.gif
```

## Interruptions and recovery

Completed algorithm runs remain usable if a later process is interrupted. Re-run individual
training with:

```powershell
.\.venv\Scripts\python.exe -m selective_marl train --algorithm selective --steps 500000 --seed 42
```

Then evaluate selected checkpoints:

```powershell
.\.venv\Scripts\python.exe -m selective_marl evaluate <checkpoint-1> <checkpoint-2>
```

