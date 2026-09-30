# Experiment protocol

Historical development protocol. The completed V2 campaigns are specified in [LIAM comparison](liam-comparison-protocol.md) and [selectivity controls](selectivity-comparison-protocol.md).

## Objective

Determine whether explicit diagnosis of semantic and behavioral shifts enables faster,
more stable adaptation than monolithic recurrent context inference.

## Unit of evaluation

A session is a sequence of standard MPE2 episodes. Environment state resets between
episodes, while an adaptive agent may retain its recurrent context. One unknown change
point is sampled per session. The first study uses episode-boundary changes; a later
stress test may introduce within-episode changes.

## Controlled factors

The benchmark is a 2 x 2 factorial design:

1. semantic mapping unchanged or permuted;
2. teammate behavioral policy unchanged or swapped.

The semantic intervention permutes only the received-message dimensions. It must not
change positions, velocities, rewards, physical actions, or random-number streams. The
first behavioral intervention rotates the physical movement selected for one teammate while
retaining the communication action and symbol convention. This controlled motor-policy
shift provides exact causal ground truth. Learned teammate-policy swaps are a later robustness
extension. Automated invariance tests are required before training.

## Information boundary

The intervention label and change point are available to the experiment harness for
scoring. They are unavailable to evaluated agents. Training may use the label as an
auxiliary target when explicitly identified as supervised diagnosis; evaluation never does.

## Baselines

1. Random policy and environment sanity checks.
2. Independent non-communicating policy.
3. MADDPG as the historical MPE baseline.
4. MAPPO.
5. Recurrent MAPPO with a GRU.
6. Channel-randomized recurrent policy.
7. Single-context adaptation without semantic/behavioral factorization.
8. Factorized model that always updates both components.
9. Proposed diagnosis with selective adaptation.
10. Oracle supplied with the intervention class.

Each baseline receives the same observation budget, evaluation sessions, change-point
distribution, and seed set. Parameter counts and training steps are reported.

## Primary hypotheses

- H1: explicit diagnosis improves intervention classification over a single latent context.
- H2: selective adaptation reduces recovery delay and cumulative regret.
- H3: selective adaptation better preserves the unaffected competence.
- H4: gains generalize to unseen symbol permutations, policies, and change points.

## Statistical protocol

- Freeze the evaluation seed list before final experiments.
- Report every seed, mean, standard deviation, median, and 95% bootstrap interval.
- Compare paired runs on identical evaluation sessions.
- Report effect sizes alongside significance tests.
- Mark exploratory analyses separately from preregistered primary comparisons.
- Never select checkpoints using the final test sessions.

## Progression gate for Simple World Comm

The second environment begins only after all of the following are true:

- the `Simple Reference` wrapper passes invariance tests;
- at least MADDPG, MAPPO, and recurrent MAPPO are reproduced;
- raw manifests can regenerate every aggregate table;
- all four intervention conditions have multiple-seed results;
- the diagnosis and selective-adaptation ablations are complete.

