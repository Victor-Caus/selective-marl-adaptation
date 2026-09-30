# LIAM comparison protocol — 2026-09-28

## Scope and provenance

User explicitly requested testing a published adaptation method after the V2
selectivity controls failed the primary criterion. Test Local Information Agent
Modelling (LIAM), Papoudakis, Christianos and Albrecht, NeurIPS 2021.

- Paper: https://arxiv.org/html/2006.09447v4
- Official source: https://github.com/uoe-agents/LIAM
- Commit: 8545b9e4237eb60ad45b7cb8ed6caec6bc4263b5, MIT license.
- Imported code: double_speaker_listener/agent.py, models.py, storage.py,
  standardise_stream.py, utils.py. External checkout remains unmodified.
- Directly retained: A2C losses and updates, independent representation optimizer,
  stop-gradient of actor loss into the encoder, LSTM, observation/action
  reconstruction, five-step chunks, GAE, running return normalization.
- Environment port changes 18 to 21 observation entries and 5+5 to 5+10 action
  entries. Resize only communication head, LSTM input, decoder communication head,
  and communication rollout storage. No imported algorithm method is replaced.

This is the official LIAM core PORTED to MPE2, not an exact reproduction of the
paper's original double-speaker benchmark or a claim about the latest SOTA.
Targeted searches did not identify an official Fastap repository; Fastap is not
included and is not replaced with a homemade imitation. Its paper/supplement
reference EPyMARL, which alone is not the Fastap implementation.

## Compatible task definition

LIAM's original formulation controls one agent against fixed partner policies.
Therefore control only agent_1 in ALL methods; agent_0 uses the SAME frozen
native-MPE2 R-MAPPO actor for that training seed. Reevaluate V2 here: old two-adapter
evaluation scores are not comparable and will not be imported into this table.
The two-agent MPE2 task, 25-step episodes, rewards, action mapping, semantic
permutation and rotation of agent_1 remain unchanged. At deployment the policy
and each V2 adapter consume their own observation/history only. Partner sampling
uses a separate seeded RNG during evaluation, identical across methods.

LIAM trains against this fixed actor with varied received-message protocols and
actuator contexts, rather than the original ten-partner population. No identity,
permutation, change point or other-agent observation is passed to LIAM.act.
Partner observations and selected actions are reconstruction targets during
training only, as in the paper. V2 retains its extra clean-symbol/change-label
training supervision; these privileges are different and must be disclosed.

## Training, fixed before final test

Five learner seeds 1..5, partner/base/memory from each corresponding completed V2
seed. LIAM starts from scratch, CPU float32, hidden 128, embedding 20,
actor LR 3e-4, representation LR 7e-4, entropy .01, value coefficient .5,
gradient norm .5, gamma .99, GAE .95, as official double-speaker config.
Use 32 environments rather than original 10 for throughput, keeping five chunks
of five steps and the original sequential update order. Episode state resets
each 25 steps; V2 semantic memory persists across episodes, as before.

Train 40,000,000 environment steps per seed (the original released configuration's
training duration). Save a prespecified earlier checkpoint at 3,420,800 steps,
matching V2's 2,998,400 actor + 422,400 memory/validation/calibration interactions
as a LEARNER-budget comparison. Partner pretraining is shared infrastructure:
LIAM additionally needs the 2,998,400-step partner, while V2's partner and own
actor were jointly trained. Thus the overall unique-interaction costs are NOT
equal. Report this; the long LIAM checkpoint receives substantially more training.
Use fixed final checkpoints, never select checkpoints from final evaluation.

Training wrapper sessions: 48 episodes, balanced none/semantic/behavioral/both,
change uniformly 10..30; independent permutations. Training world seeds start at
20,000,000 plus learner seed*200,000 and rank*1,000. No test-seed tuning.

## Paired evaluation and inference

Seven modes: frozen, motor_only, joint_v2, unguarded_v2, selective_v2,
liam_matched, liam_long. Five training seeds, four conditions, twenty sessions
per condition/model, sixty episodes, change uniformly 12..35. New final world
seeds 16,000,001+1,000*i. Smoke seeds 18,000,001 are separate. Actions stochastic.
For LIAM the encoder state resets each episode, matching the released method;
no invented cross-episode variant is labeled original LIAM.

Report seed means of pre/post returns and post-minus-pre change, and paired
V2-minus-LIAM differences with Student-t 95% intervals (n=5, limited power).
Primary comparator liam_long: three changed conditions, one-sided superiority
tests, Holm adjustment over three. No-change noninferiority margin .25 using
one-sided 95% bound. Claim this restricted criterion only if all three corrected
p<.05 with positive mean and normal lower bound>-.25. Secondary comparisons
including liam_matched are descriptive. Always show raw pre-shift competence:
post-return differences alone do not isolate the speed/quality of adaptation.
Do not infer general SOTA superiority from this single ported-task comparison.

## Validation and operations

Validate original-dimension action/value equivalence to imported A2C, gradient
isolation, expanded action storage, then train/evaluate/report a disjoint smoke.
Do not launch final evaluation if any training fails/nonfinite parameters occur.
Capture source/dependencies/config/checkpoint hashes and original source revision.
Results: results/liam-comparison-20260928-r1. tmux: marl-liam-comparison.
No automatic retries or changes after final results are opened.
