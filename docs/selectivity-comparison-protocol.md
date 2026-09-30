# Selectivity comparison — frozen protocol, 2026-09-28

## Question and scope

Does V2's selective routing improve on simpler adaptation rules, rather than
merely improve on a policy that never adapts? Negative outcomes are valid.
Do not tune thresholds, architecture or checkpoints on this campaign's test set.

This campaign contains the published R-MAPPO optimizer (pinned official source
de66d7a4b23fac2513f56f96f73b3f5cb96695ac) and project-specific simple controls.
It is NOT a reproduction of Fastap, LIAM, ECTL, or proof of superiority over
the latest specialized adaptation literature. Those comparisons remain pending.
Fastap models changing teammate populations; our motor perturbation is only an
actuator rotation. Matching these problem definitions requires a separate port.

Primary sources consulted:
- R-MAPPO: https://arxiv.org/abs/2103.01955
- Fastap: https://proceedings.mlr.press/v216/zhang23a.html
- LIAM official implementation: https://github.com/uoe-agents/LIAM
- Channel randomization: https://arxiv.org/abs/2104.09557
- ECTL: https://www.ijcai.org/proceedings/2024/5

## Matched ablations

All V2 variants use exactly the same actor, learned memory weights, local inputs,
motor system identification, and per-seed training data. No retraining is needed
to change a routing rule. Names and differences from selective_v2:

| Mode | Difference |
|---|---|
| frozen | Original R-MAPPO actor, no adaptation |
| motor_only | Only the simple online motor identification/remapping |
| unguarded_v2 | Semantic correction always active after 8 episodes, no rollback |
| joint_v2 | Also enables semantic correction on motor detection |
| reward_only_v2 | Removes semantic confidence from the activation rule |
| confidence_only_v2 | Removes reward degradation from the activation rule |
| no_rollback_v2 | Removes rollback protection; identical selective activation |
| shared_stream_v2 | Same gate, but adapted GRU stream drives both action heads |
| selective_v2 | Unchanged V2 including sender isolation and rollback |

These isolate routing/protection/sender isolation, not every possible monolithic
architecture. None of the rules is described as a published SOTA implementation.

## Additional training controls

For each seed, warm-start the same official R-MAPPO actor, train all actor weights
with the pinned optimizer, and initialize a fresh critic/ValueNorm (original
checkpoints did not persist normalizer statistics). Both controls use LR 1e-4,
128 environments, horizon 25, 15 PPO epochs, and exactly 422,400 extra environment
steps (132 updates). No test-time weight updates or oracle labels.

- rmappo_continued: train on unchanged environments.
- rmappo_randomized: 48-episode training sessions; cycle four conditions, change
  episode uniformly 10..30, random message permutation. The distribution mirrors
  memory collection. This is our domain-randomization control using official
  R-MAPPO, not an exact reproduction of the channel-randomization paper.

V2 used 307,200 training + 38,400 validation + 76,800 normal calibration steps.
The two R-MAPPO controls spend the entire 422,400 budget on optimization. Thus
total interaction budget is equal, allocation and supervision are not identical.
V2 has clean-symbol/change-label training supervision; controls do not. V2 has
cross-episode memory; standard R-MAPPO resets recurrence each episode. Report
these differences. The matched V2 ablations are the causal test of selectivity.

## Seeds and evaluation

Five independent training seeds 1..5. Reuse completed base/memory checkpoints
1..3 with hashes; train base/memory 4..5 by the unchanged V2 recipe (2,998,400
base steps and the memory budget above). Do not retrain or replace existing V2.

All 11 methods: none, semantic, behavioral, both; 20 paired sessions per training
seed/condition, 60 episodes each, change episode uniformly 12..35. Evaluation
seeds 12000001 + 1000*i, i=0..19, disjoint from previous final evaluations and
training/validation/calibration. Environment seeds and sampling seeds match
across methods, though diverging policies cannot have identical trajectories.
Evaluation uses stochastic actions and the existing native MPE2 mean reward.

## Analysis decided before evaluation

Aggregate sessions within each training seed. Episodes/sessions are not treated
as independent training replications. Report per-condition post-change return,
common-frozen-baseline regret, pre-change return, and normal-session activation
and actual action-intervention rates (always-on activation is not a false detector
alarm). Compare the selective-minus-control return using paired seed differences.

Primary efficacy endpoint: equally weighted mean post-change return over semantic,
behavioral, both. Primary comparators: motor_only, joint_v2, unguarded_v2.
Report two-sided Student-t 95% intervals over five paired seed means, and
one-sided paired t-tests for superiority with Holm adjustment across these three
primary comparisons. Five seeds give limited power/normality assessment.
The primary safety endpoint is no-change return difference for each primary
control, with a prespecified pragmatic noninferiority margin of 0.25 return units
(about 3% of the historical normal return magnitude). Require lower one-sided
95% t bound > -0.25 for all three controls. This margin is a design choice, not
a field-standard validated threshold. Secondary comparisons are descriptive.

Only call the primary criterion met if all three adjusted efficacy p-values are
< .05 with positive differences AND all three normal noninferiority checks pass.
Do not equate this criterion with universal superiority or publication readiness.
Keep per-condition failures visible even when an aggregate passes.

## Operations

Run sequentially, fail fast, never overwrite output directories. Capture source
snapshot, SHA256s, dependency freeze, original git revision and exact commands.
Run meaningful ablation/wrapper tests and a full pipeline smoke test before launch.
Results root: results/selectivity-controls-20260928-r1.
tmux: marl-selectivity-controls. No automatically restarting failed stages.
