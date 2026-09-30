#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.external/envs/selective-marl-linux/bin/python
OUT="${1:?Pass a new output directory}"
if [[ -e "$OUT" ]]; then echo "Output already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
trap 'printf "failed line=%s\n" "$LINENO" > "$OUT/status.txt"' ERR
echo testing > "$OUT/status.txt"
"$PY" -m pytest -q > "$OUT/tests.log" 2>&1
"$PY" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/base-commit.txt"
git diff > "$OUT/tracked-changes.diff"
tar -czf "$OUT/source-snapshot.tar.gz" src scripts tests docs reproductions/references.lock.json
sha256sum "$OUT/source-snapshot.tar.gz" > "$OUT/source-sha256.txt"
cp docs/selectivity-comparison-protocol.md "$OUT/PROTOCOL.md"
# Record actual commands with arguments in the launcher log.
set -x
for SEED in 1 2 3 4 5; do
  if [[ "$SEED" == 1 ]]; then BASE=results/backend-full-s1-20260927
  elif [[ "$SEED" -le 3 ]]; then BASE="results/adaptation-v2-20260928-r1/base-s$SEED"
  else
    BASE="$OUT/base-s$SEED"
    echo "base_training seed=$SEED" > "$OUT/status.txt"
    "$PY" -u scripts/train-mpe2-reference-backend.py --output "$BASE" \
      --steps 3000000 --envs 128 --seed "$SEED" --device cuda > "$OUT/base-s$SEED.log" 2>&1
  fi
  "$PY" - "$BASE/metadata.json" "$SEED" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m['status'] == 'complete' and m['actual_environment_steps'] == 2998400
assert m['training_seed'] == int(sys.argv[2])
PY
  if [[ "$SEED" -le 3 ]]; then MEMORY="results/adaptation-v2-20260928-r1/memory-s$SEED"
  else
    MEMORY="$OUT/memory-s$SEED"
    echo "memory_training seed=$SEED" > "$OUT/status.txt"
    "$PY" -u scripts/train-semantic-memory.py --actor "$BASE/models/actor.pt" \
      --output "$MEMORY" --seed "$SEED" --sessions 256 --validation-sessions 32 \
      --calibration-sessions 64 --episodes 48 --epochs 20 --device cuda > "$OUT/memory-s$SEED.log" 2>&1
  fi
  "$PY" - "$MEMORY/status.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))['status'] == 'complete'
PY
  sha256sum "$BASE/models/actor.pt" "$MEMORY/memory.pt" >> "$OUT/reused-checkpoints-sha256.txt"
  for CONTROL in continued randomized; do
    EXTRA=()
    if [[ "$CONTROL" == randomized ]]; then EXTRA=(--randomize); fi
    echo "control_training seed=$SEED control=$CONTROL" > "$OUT/status.txt"
    "$PY" -u scripts/train-mpe2-reference-backend.py --initial-actor "$BASE/models/actor.pt" \
      --output "$OUT/$CONTROL-s$SEED" --steps 422400 --envs 128 --seed "$SEED" \
      --learning-rate 0.0001 --device cuda "${EXTRA[@]}" > "$OUT/$CONTROL-s$SEED.log" 2>&1
    echo "control_evaluation seed=$SEED control=$CONTROL" > "$OUT/status.txt"
    "$PY" -u scripts/evaluate-online-adapters.py --actor "$OUT/$CONTROL-s$SEED/models/actor.pt" \
      --output "$OUT/$CONTROL-eval-s$SEED" --sessions 20 --episodes 60 \
      --change-range 12 35 --seed 12000001 --modes frozen > "$OUT/$CONTROL-eval-s$SEED.log" 2>&1
  done
  echo "ablation_evaluation seed=$SEED" > "$OUT/status.txt"
  "$PY" -u scripts/evaluate-online-adapters.py --actor "$BASE/models/actor.pt" \
    --memory "$MEMORY/memory.pt" --output "$OUT/variants-s$SEED" \
    --sessions 20 --episodes 60 --change-range 12 35 --seed 12000001 \
    --modes frozen motor_only unguarded_v2 joint_v2 reward_only_v2 confidence_only_v2 \
      no_rollback_v2 shared_stream_v2 selective_v2 > "$OUT/variants-s$SEED.log" 2>&1
done
echo summarizing > "$OUT/status.txt"
"$PY" scripts/summarize-selectivity-controls.py "$OUT"
echo complete > "$OUT/status.txt"
