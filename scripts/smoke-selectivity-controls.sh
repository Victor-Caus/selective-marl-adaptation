#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.external/envs/selective-marl-linux/bin/python
OUT="${1:?Pass a new smoke directory}"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'printf "failed line=%s\n" "$LINENO" > "$OUT/status.txt"' ERR
"$PY" -m pytest -q > "$OUT/tests.log" 2>&1
BASE=results/backend-full-s1-20260927/models/actor.pt
MEMORY=results/adaptation-v2-20260928-r1/memory-s1/memory.pt
for CONTROL in continued randomized; do
  EXTRA=()
  if [[ "$CONTROL" == randomized ]]; then EXTRA=(--randomize); fi
  echo "training $CONTROL" > "$OUT/status.txt"
  "$PY" -u scripts/train-mpe2-reference-backend.py --initial-actor "$BASE" \
    --output "$OUT/$CONTROL-s1" --steps 3200 --envs 4 --seed 81 \
    --learning-rate 0.0001 --device cuda "${EXTRA[@]}" > "$OUT/$CONTROL-s1.log" 2>&1
  echo "evaluating $CONTROL" > "$OUT/status.txt"
  "$PY" scripts/evaluate-online-adapters.py --actor "$OUT/$CONTROL-s1/models/actor.pt" \
    --output "$OUT/$CONTROL-eval-s1" --sessions 1 --episodes 60 \
    --change-range 12 35 --seed 14000001 --modes frozen > "$OUT/$CONTROL-eval-s1.log" 2>&1
done
echo variants > "$OUT/status.txt"
"$PY" scripts/evaluate-online-adapters.py --actor "$BASE" --memory "$MEMORY" \
  --output "$OUT/variants-s1" --sessions 1 --episodes 60 --change-range 12 35 \
  --seed 14000001 --modes frozen motor_only unguarded_v2 joint_v2 reward_only_v2 \
    confidence_only_v2 no_rollback_v2 shared_stream_v2 selective_v2 > "$OUT/variants-s1.log" 2>&1
"$PY" scripts/summarize-selectivity-controls.py "$OUT" --seeds 1 --sessions 1
echo complete > "$OUT/status.txt"
