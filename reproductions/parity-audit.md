# Initial parity audit

This audit compares the preliminary implementation with the pinned reference sources. Line
claims refer to the immutable revisions in `references.lock.json`, not to moving branches.

| Dimension | Pinned reference | Preliminary implementation | Required action |
| --- | --- | --- | --- |
| Environment | OpenAI MPE `simple_reference.py` | MPE2 1.1.1 `simple_reference_v3` | Establish transition and reward parity |
| Action distribution | Two categorical heads for MPE `MultiDiscrete(5, 10)` in official MAPPO | One categorical head over 50 products | Implement a factorized baseline |
| Shared reward | Historical MPE sums both agent rewards when `world.collaborative` is true | MPE2 with `local_ratio=0.5` | Reproduce both reward definitions and document scale |
| MAPPO budget | 3,000,000 environment steps | Quick: 20,000; planned research: 500,000 | Add the official-budget reproduction profile |
| Rollout collection | 128 rollout environments | One sequential environment | Match the reference collector before comparing curves |
| PPO updates | 15 epochs, one minibatch | Quick: 4 epochs; research: 8 | Match the reference optimizer schedule |
| MAPPO learning rates | Actor and critic `7e-4` | Actor and critic `3e-4` | Add an immutable reference configuration |
| Output initialization | Gain `0.01`, orthogonal initialization enabled by default | PyTorch default linear initialization | Match initialization |
| Value normalization | Enabled by default | Not implemented | Add parity implementation or use official runner |
| Recurrent training | Sequence-aware recurrent generator | Flattened transitions with stored hidden states | Use the official recurrent batching semantics |
| MADDPG historical budget | 60,000 episodes, horizon 25 | Quick: 400 episodes | Do not compare current curve with the historical code |
| MADDPG runtime | Python 3.5.4, Gym 0.10.5, TensorFlow 1.8.0 | Python 3.11, MPE2, PyTorch 2.14 | Run historical code only in an isolated legacy runtime |

## Source locations

- MAPPO settings: `marlbenchmark-on-policy/onpolicy/scripts/train_mpe_scripts/train_mpe_reference.sh`
- MAPPO factorized action heads: `marlbenchmark-on-policy/onpolicy/algorithms/utils/act.py`
- Historical reward sharing and action space: `openai-mpe/multiagent/environment.py`
- Historical task definition: `openai-mpe/multiagent/scenarios/simple_reference.py`
- Historical MADDPG defaults: `openai-maddpg/experiments/train.py` and its `README.md`

The source paths above are relative to `.external/reference-sources/` after running the fetch
script. They are evidence for configuration choices, not files copied into this repository.
