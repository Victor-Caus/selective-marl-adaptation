# Baseline reproduction plan

## Purpose

Validate the environment and learning implementations before evaluating semantic or
behavioral shifts. The current in-house algorithms are preliminary reimplementations, not
validated substitutes for the cited baselines.

## Experiment ladder

1. Run each pinned reference implementation on its unmodified environment and configuration.
2. Record dependencies, action representation, reward construction, horizon, seeds, training
   budget, checkpoint rule, and raw learning curves.
3. Match those choices in the in-house implementation and compare learning curves without
   interventions.
4. Freeze a validated baseline configuration.
5. Add semantic-only, behavioral-only, joint, and no-change interventions without changing
   the learner.
6. Add a conventional recurrent baseline with one context.
7. Add factorized contexts with oracle routing.
8. Add learned diagnosis and selective adaptation.

Only one experimental factor changes between adjacent stages.

## Known parity differences to resolve

- MPE2 currently runs with `local_ratio=0.5`; the historical MPE environment used its own
  collaborative reward construction.
- MPE2 exposes the discrete action as the Cartesian product of five movement choices and ten
  communication symbols. The preliminary policies use one 50-class categorical head, while
  reference implementations may factor the two components or use continuous actions.
- The preliminary PPO implementation omits some features from the official MAPPO codebase,
  including its exact recurrent batching and value normalization path.
- A 20,000-step quick run is not comparable to historical runs with much larger budgets.

The pinned official MAPPO script for `simple_reference` uses 3,000,000 environment steps,
128 rollout environments, 15 PPO epochs, one minibatch, a 25-step horizon, actor and critic
learning rates of `7e-4`, output gain `0.01`, and its default value normalization. These are
the first settings to reproduce; they are not approximated by the existing `quick` profile.

These differences must be eliminated or explicitly measured before comparing scores.

## Executable Windows reference path

The official R-MAPPO source can be installed and run in an isolated project-local environment:

```powershell
.\scripts\setup-reference-mappo.ps1
.\scripts\run-reference-mappo.ps1 -Profile smoke
.\scripts\run-reference-mappo.ps1 -Profile pilot
.\scripts\run-reference-mappo.ps1 -Profile full
```

Raw outputs, dependency snapshots, console logs, normalized CSV metrics, a report, and a
learning-curve plot are written below ignored `results/reference-runs/`. The precise Windows
compatibility choices, failed 128/32-process scaling checks, and the sequential 128-environment
adapter are recorded in `reproductions/host-deviations.md`.

## Acceptance gates

- A trained reference policy must clearly outperform a random policy.
- Communication must provide a reproducible advantage over a no-communication ablation.
- The in-house baseline must track the reference learning curve across at least three seeds.
- Metrics must use equivalent reward scale, episode horizon, evaluation policy, and checkpoint
  selection.
- Every result row must link to a source revision, resolved configuration, seed, and raw run.

If a paper does not report `simple_reference`, the result is labeled a repository/configuration
reproduction rather than a reproduction of a published number.
