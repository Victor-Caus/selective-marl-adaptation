# Reproducibility checklist

Each experiment run must record:

- Git commit and dirty-tree status;
- complete resolved configuration;
- Python, operating system, MPE2, PettingZoo, NumPy, and framework versions;
- accelerator and deterministic-algorithm settings;
- training, environment, intervention, and evaluation seeds;
- algorithm and environment steps;
- checkpoint-selection rule;
- wall-clock time and termination reason;
- raw episode-level metrics;
- intervention kind, change point, and permutation or policy identifiers.

Aggregate artifacts must be generated from immutable raw records. Scripts should reject
missing seeds, duplicate run identifiers, and mixed protocol versions. Figures used in the
paper must be reproducible from a committed command and must not be manually edited.

