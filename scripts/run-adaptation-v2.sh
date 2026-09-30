#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=.external/envs/selective-marl-linux/bin/python
OUT="${1:?Pass a new output directory}"
if [[ -e "$OUT" ]]; then echo "Output already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
trap 'printf "failed line=%s\n" "$LINENO" > "$OUT/status.txt"' ERR
printf 'testing\n' > "$OUT/status.txt"
"$PY" -m pytest -q > "$OUT/tests.log" 2>&1
"$PY" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/base-commit.txt"
git diff > "$OUT/tracked-changes.diff"
tar -czf "$OUT/source-snapshot.tar.gz" src scripts tests docs/research-ledger.md reproductions/references.lock.json
sha256sum "$OUT/source-snapshot.tar.gz" > "$OUT/source-sha256.txt"
cp docs/research-ledger.md "$OUT/PROTOCOL.md"
for SEED in 1 2 3; do
  if [[ "$SEED" == 1 ]]; then
    BASE=results/backend-full-s1-20260927
    "$PY" - "$BASE/metadata.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m['status'] == 'complete' and m['actual_environment_steps'] == 2998400
assert m['training_seed'] == 1
PY
  else
    BASE="$OUT/base-s$SEED"
    echo "base_training seed=$SEED" > "$OUT/status.txt"
    "$PY" -u scripts/train-mpe2-reference-backend.py --output "$BASE" \
      --steps 3000000 --envs 128 --seed "$SEED" --device cuda > "$OUT/base-s$SEED.log" 2>&1
  fi
  sha256sum "$BASE/models/actor.pt" >> "$OUT/base-checkpoints-sha256.txt"
  echo "memory_training seed=$SEED" > "$OUT/status.txt"
  "$PY" -u scripts/train-semantic-memory.py --actor "$BASE/models/actor.pt" \
    --output "$OUT/memory-s$SEED" --seed "$SEED" --sessions 256 \
    --validation-sessions 32 --calibration-sessions 64 --episodes 48 --epochs 20 \
    --device cuda > "$OUT/memory-s$SEED.log" 2>&1
  echo "evaluation seed=$SEED" > "$OUT/status.txt"
  "$PY" -u scripts/evaluate-online-adapters.py --actor "$BASE/models/actor.pt" \
    --memory "$OUT/memory-s$SEED/memory.pt" --output "$OUT/evaluation-s$SEED" \
    --sessions 5 --episodes 60 --change-range 12 35 --seed 9000001 \
    --modes frozen selective motor_only unguarded_v2 joint_v2 selective_v2 \
    > "$OUT/evaluation-s$SEED.log" 2>&1
done
"$PY" scripts/summarize-adaptation-v2.py "$OUT"
echo complete > "$OUT/status.txt"
