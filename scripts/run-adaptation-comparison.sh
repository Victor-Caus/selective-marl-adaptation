#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export MPLBACKEND=Agg
PY=.external/envs/selective-marl-linux/bin/python
OUT=results/comparison-20260927-v1
mkdir -p "$OUT"
trap 'echo failed > "$OUT/status.txt"' ERR
echo validating > "$OUT/status.txt"
"$PY" -m pytest tests/test_reference_backend.py tests/test_metrics.py -q > "$OUT/tests.log" 2>&1
"$PY" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/base-commit.txt"
sha256sum src/selective_marl/adapters.py src/selective_marl/environments/reference_backend.py \
  src/selective_marl/evaluation/metrics.py scripts/evaluate-online-adapters.py \
  scripts/train-mpe2-reference-backend.py > "$OUT/source-sha256.txt"
tar -czf "$OUT/source-snapshot.tar.gz" src/selective_marl/adapters.py \
  src/selective_marl/environments/reference_backend.py src/selective_marl/evaluation/metrics.py \
  scripts/evaluate-online-adapters.py scripts/train-mpe2-reference-backend.py \
  scripts/run-adaptation-comparison.sh tests/test_reference_backend.py tests/test_metrics.py

echo baseline_legacy > "$OUT/status.txt"
"$PY" -u scripts/evaluate-online-adapters.py \
  --legacy-checkpoint results/runs/20260927T155340Z-train-rmappo-seed42/checkpoint.pt \
  --output "$OUT/legacy-common" --sessions 5 --episodes 60 --change-episode 20 \
  --seed 20001 --modes frozen --deterministic > "$OUT/legacy-common.log" 2>&1

echo baseline_reference_transfer > "$OUT/status.txt"
"$PY" -u scripts/evaluate-online-adapters.py \
  --actor results/reference-runs/linux_full_s1_20260927T150848Z/official-output/run1/models/actor.pt \
  --output "$OUT/reference-common" --sessions 5 --episodes 60 --change-episode 20 \
  --seed 20001 --modes frozen --deterministic > "$OUT/reference-common.log" 2>&1

echo waiting_for_training > "$OUT/status.txt"
"$PY" - <<'PY'
import json, time
from pathlib import Path
p = Path('results/backend-full-s1-20260927/metadata.json')
for _ in range(120):
    state = json.loads(p.read_text())['status']
    if state == 'complete':
        break
    if state == 'failed':
        raise RuntimeError('Base training failed')
    time.sleep(15)
else:
    raise TimeoutError('Base training did not finish within 30 minutes')
PY

echo baseline_new > "$OUT/status.txt"
"$PY" -u scripts/evaluate-online-adapters.py \
  --actor results/backend-full-s1-20260927/models/actor.pt \
  --output "$OUT/new-common" --sessions 5 --episodes 60 --change-episode 20 \
  --seed 20001 --modes frozen --deterministic > "$OUT/new-common.log" 2>&1

echo adaptation_validation > "$OUT/status.txt"
"$PY" -u scripts/evaluate-online-adapters.py \
  --actor results/backend-full-s1-20260927/models/actor.pt \
  --output "$OUT/adaptation-validation" --sessions 3 --episodes 60 --change-episode 20 \
  --seed 5001 --modes frozen joint selective > "$OUT/adaptation-validation.log" 2>&1
echo complete > "$OUT/status.txt"
