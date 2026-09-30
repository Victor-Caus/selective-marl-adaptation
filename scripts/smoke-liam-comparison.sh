#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=.external/envs/selective-marl-linux/bin/python
OUT="${1:?New smoke output required}"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'printf "failed line=%s\n" "$LINENO" > "$OUT/status.txt"' ERR
echo tests > "$OUT/status.txt"
"$PY" -m pytest -q > "$OUT/tests.log" 2>&1
echo training > "$OUT/status.txt"
BASE=results/backend-full-s1-20260927/models/actor.pt
"$PY" -u scripts/train-liam-reference.py --actor "$BASE" --output "$OUT/train-s1" \
  --seed 81 --steps 3200 --matched-steps 1600 --envs 32 > "$OUT/train-s1.log" 2>&1
echo evaluation > "$OUT/status.txt"
"$PY" scripts/evaluate-liam-comparison.py --actor "$BASE" \
  --memory results/adaptation-v2-20260928-r1/memory-s1/memory.pt \
  --matched "$OUT/train-s1/matched.pt" --long "$OUT/train-s1/final.pt" \
  --output "$OUT/evaluation-s1" --sessions 1 --seed 18000001 > "$OUT/evaluation-s1.log" 2>&1
"$PY" scripts/summarize-liam-comparison.py "$OUT" --seeds 1 --sessions 1 --steps 3200
echo complete > "$OUT/status.txt"
