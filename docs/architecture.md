# Architecture

## Environment boundary

`ReferenceSessionEnv` wraps the official MPE2 parallel environment. A semantic intervention
permutes only the final ten received-message dimensions. A behavioral intervention rotates
only the movement component of `agent_1`'s discrete product action and preserves its message
index. Tests verify both invariants.

The behavioral transform is a controlled motor-policy shift, not a claim that every real
teammate change is a rotation. It provides a known causal source for the first benchmark.
Additional learned-policy swaps can be added after the controlled study is stable.

## Learning systems

All PPO variants use a shared decentralized actor. Their critic receives the concatenated
observations of both agents during training. The feed-forward actor is the MAPPO baseline;
R-MAPPO adds persistent GRU state across episodes in a session.

The selective model splits each observation into eleven physical dimensions and ten message
dimensions. Separate encoders and GRU contexts track behavioral and semantic evidence. A
four-class auxiliary head predicts no change, semantic change, behavioral change, or both.
Its predicted class freezes the context believed to remain valid and updates only the
affected context. Training uses the simulator label for routing; evaluation reports both
predicted routing and an oracle-gated upper-bound ablation.

MADDPG uses two decentralized discrete actors, one centralized critic per agent, target
networks, experience replay, epsilon exploration, and soft target updates.

## Evidence flow

Raw transitions become episode records, session-level disruption/recovery metrics, aggregate
tables, and figures. The experiment label is retained by the harness for scoring but is not
included in the observation. All generated evidence includes a configuration, Git commit,
dependency metadata, and seed.

