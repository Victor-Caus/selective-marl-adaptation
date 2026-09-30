# Architecture

## Environment

`ReferenceSessionEnv` wraps MPE2 Simple Reference. Received-message interventions permute the final ten observation entries at both receivers. The movement intervention rotates only agent 1's movement command, preserving its outgoing symbol. Intervention labels and change points remain outside learner observations.

## V2

The base is a frozen recurrent MAPPO actor. A separate 128-unit GRU estimates received symbols in the original code from local observations, previous actions and rewards. This receiver memory persists across episodes within a session. A reward/confidence gate enables message correction; a trial can withdraw it if performance deteriorates.

The actor is evaluated twice with shared weights and independent recurrent states: corrected observations supply movement probabilities; raw observations supply outgoing-message probabilities. Both actor states reset each episode. The receiver GRU does not share weights with this actor.

The motor module estimates command effects from local velocity transitions and applies an inverse mapping to movement probabilities. It is not another neural policy. Evaluation updates memories, detector statistics and this mapping, but no neural weights.

## LIAM comparison

The port uses the released LIAM actor, reconstruction objective and local context encoder. The policy loss does not update the representation encoder. Training uses partner observations/actions as reconstruction targets; execution uses local history. Dimensions and the environment interface are adapted to this task.

All methods control only agent 1 with the same frozen partner per training seed. This differs from the earlier simpler-control campaign, where both agents use adapters. Their scores must be read separately.

## Earlier implementations

The repository also retains preliminary in-house MAPPO, MADDPG and factorized-context policies used during development. These are not the V2 architecture and are not substitutes for the reference-backed comparisons in the current paper.

## Records

Episode returns feed session summaries and then training-seed aggregates. The environment records true interventions for scoring, not as learner inputs. Configurations, source revisions and checkpoint hashes identify each campaign.
