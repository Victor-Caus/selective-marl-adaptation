# Related work map

This document records the closest prior work and the boundary of the intended contribution.
It is a living map, not a claim that the literature search is exhaustive.

## Environment and learning foundations

- Mordatch, I., and Abbeel, P. (2017). *Emergence of Grounded Compositional Language in
  Multi-Agent Populations*. Introduces communication-oriented tasks underlying MPE.
- Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
  Environments*. Introduces MADDPG and releases the original MPE environments.
- Yu, C. et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*.
  Establishes MAPPO as a strong cooperative MARL baseline.

## Protocol variation and unfamiliar partners

- Cope, D., and Schoots, N. (2021). *Learning to Communicate with Strangers via Channel
  Randomisation Methods*. Studies message mutation and channel permutation for zero-shot
  communication.
- Hu, H. et al. (2020). *Other-Play for Zero-Shot Coordination*. Uses environment symmetries
  to improve coordination with independently trained partners.
- Cope, D., and McBurney, P. (2024). *Learning Translations: Emergent Communication
  Pretraining for Cooperative Language Acquisition*. Learns translation into a target
  community's protocol from interaction data.

## Behavioral non-stationarity

- Ravula, M. et al. (2019). *Ad Hoc Teamwork With Behavior Switching Agents*. Detects
  teammate type changes during cooperation.
- Zhang, Z. et al. (2023). *Fast Teammate Adaptation in the Presence of Sudden Policy Change*.
  Adapts to teammate policy changes within an episode using inferred context.

## Factorization, robustness, and measurement

- Park, B., and Choi, J. (2024). *Message Action Adapter Framework in Multi-Agent
  Reinforcement Learning*. Separates an observation-conditioned base action from a
  message-conditioned residual adapter.
- Yu, L. et al. (2024). *Robust Communicative Multi-Agent Reinforcement Learning with Active
  Defense*. Estimates message reliability and reduces harmful message influence.
- Lowe, R. et al. (2019). *On the Pitfalls of Measuring Emergent Communication*. Shows why
  message/action correlation and reward improvement alone do not establish causal influence.

## Intended contribution boundary

Prior work covers protocol randomization, translation, sudden teammate-policy changes,
message/action factorization, and communication robustness. The working contribution is the
combination of:

1. controlled semantic-only, behavioral-only, joint, and no-change interventions;
2. online attribution of the coordination failure to its mechanism;
3. routing adaptation to only the inferred affected component;
4. explicit measurement of recovery and preservation of unaffected competence.

The eventual manuscript should use cautious language such as “we did not identify a prior
evaluation combining these elements” until backward and forward citation searches are
completed immediately before submission.

## Primary links

- MPE2: https://mpe2.farama.org/
- MADDPG: https://arxiv.org/abs/1706.02275
- MAPPO: https://arxiv.org/abs/2103.01955
- Channel randomisation: https://arxiv.org/abs/2104.09557
- Fastap: https://proceedings.mlr.press/v216/zhang23a.html
- Learning Translations: https://www.ijcai.org/proceedings/2024/5
- MAAF: https://doi.org/10.3390/app142110079
- ADMAC: https://doi.org/10.1609/aaai.v38i16.29708
- Communication measurement: https://ai.meta.com/research/publications/on-the-pitfalls-of-measuring-emergent-communication/

