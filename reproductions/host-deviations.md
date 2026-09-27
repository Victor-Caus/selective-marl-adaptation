# MAPPO host deviations

The pinned `simple_reference` script targets a Linux/CUDA-era environment and launches 128
rollout processes. The current reproduction host is Windows and CPU-only. The algorithm,
environment source, reward definition, horizon, optimizer settings, recurrent policy, action
factorization, seed, and total environment-step budget remain those of the pinned source.

## Compatibility choices

- Python 3.8.20 replaces the repository's archived Python 3.6.2 environment.
- PyTorch 1.13.1 CPU replaces PyTorch 1.5.1 CUDA because the archived GPU stack is not
  available on this host.
- TensorBoardX 2.6.2.2 replaces 2.0 because 2.0 cannot create nested scalar directories on
  Windows. This affects logging only.
- A short NTFS junction points to the pinned checkout to avoid the Windows path-length limit.
  It does not copy or edit the source.
- Tests with 128 and 32 rollout processes exhausted Windows virtual memory before the first
  environment step. The Windows adapter therefore places the same 128 ranked, independently
  seeded environments in the official `DummyVecEnv` and advances them sequentially in one
  process. This preserves the official 3,200-step rollout batch and approximately 937 PPO
  updates over 3,000,000 environment steps. A discarded 16-process pilot would instead have
  performed 7,500 smaller-batch updates and is not used as a reference result.

Environment process placement and package versions still differ, so the local run is a
host-feasible reference reproduction, not a bitwise or exact paper reproduction. An exact
128-process Linux run remains a later validation, but the local adapter retains the official
batch size, update count, seeds, algorithm, and environment logic.
