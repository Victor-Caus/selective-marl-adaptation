# Reference reproductions

This directory defines how external baselines are obtained and compared before any proposed
adaptation mechanism is introduced. External repositories are not vendored. The fetch script
clones them into the ignored `.external/reference-sources/` directory and checks out the exact
commits recorded in `references.lock.json`.

## Provenance rule

Every external implementation, configuration, equation, or result used by this project must
have a source URL, immutable revision or publication identifier, license, citation key, and a
short description of how it was used. Modified external code must retain its original license
and be represented as a documented patch rather than unattributed copied code.

Fetch the pinned sources on Windows with:

```powershell
.\scripts\fetch-reference-sources.ps1
```

Audit the local prerequisites and checked-out commits with:

```powershell
.\scripts\check-reproduction-prereqs.ps1
```

Fetching is not the same as reproducing. The historical OpenAI MADDPG code requires Python
3.5.4, TensorFlow 1.8.0, Gym 0.10.5, and NumPy 1.14.5. It still requires a separate
TensorFlow 1.x-compatible environment. The official MAPPO code has a project-local
micromamba setup and Windows runner:

```powershell
.\scripts\setup-reference-mappo.ps1
.\scripts\run-reference-mappo.ps1 -Profile smoke
```

The MAPPO runner records every compatibility choice and must not be described as an exact
paper reproduction when its rollout-process count differs from the pinned Linux script.
No compatibility edit may be called an exact reproduction.

See `docs/reproduction-plan.md` for the experiment ladder and acceptance gates.
The concrete differences found in the first source audit are recorded in `parity-audit.md`.
