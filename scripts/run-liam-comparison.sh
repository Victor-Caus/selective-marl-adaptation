#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=.external/envs/selective-marl-linux/bin/python
OUT="${1:?New output directory required}"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'printf "failed line=%s\n" "$LINENO" > "$OUT/status.txt"' ERR
echo testing > "$OUT/status.txt"
"$PY" -m pytest -q > "$OUT/tests.log" 2>&1
"$PY" -m pip freeze > "$OUT/environment.txt"
git -C .external/reference-sources/liam rev-parse HEAD > "$OUT/liam-commit.txt"
git rev-parse HEAD > "$OUT/base-commit.txt"
git diff > "$OUT/tracked-changes.diff"
tar -czf "$OUT/source-snapshot.tar.gz" src scripts tests docs reproductions/references.lock.json
sha256sum "$OUT/source-snapshot.tar.gz" > "$OUT/source-sha256.txt"
cp docs/liam-comparison-protocol.md "$OUT/PROTOCOL.md"
set -x
for SEED in 1 2 3 4 5; do
  if [[ "$SEED" == 1 ]]; then BASE=results/backend-full-s1-20260927
  elif [[ "$SEED" -le 3 ]]; then BASE="results/adaptation-v2-20260928-r1/base-s$SEED"
  else BASE="results/selectivity-controls-20260928-r1/base-s$SEED"; fi
  if [[ "$SEED" -le 3 ]]; then MEMORY="results/adaptation-v2-20260928-r1/memory-s$SEED/memory.pt"
  else MEMORY="results/selectivity-controls-20260928-r1/memory-s$SEED/memory.pt"; fi
  sha256sum "$BASE/models/actor.pt" "$MEMORY" >> "$OUT/partner-memory-sha256.txt"
  echo "liam_training seed=$SEED" > "$OUT/status.txt"
  "$PY" -u scripts/train-liam-reference.py --actor "$BASE/models/actor.pt" \
    --output "$OUT/train-s$SEED" --seed "$SEED" --steps 40000000 --envs 32 \
    --matched-steps 3420800 > "$OUT/train-s$SEED.log" 2>&1
  sha256sum "$OUT/train-s$SEED/matched.pt" "$OUT/train-s$SEED/final.pt" >> "$OUT/liam-checkpoints-sha256.txt"
  echo "evaluation seed=$SEED" > "$OUT/status.txt"
  "$PY" -u scripts/evaluate-liam-comparison.py --actor "$BASE/models/actor.pt" \
    --memory "$MEMORY" --matched "$OUT/train-s$SEED/matched.pt" --long "$OUT/train-s$SEED/final.pt" \
    --output "$OUT/evaluation-s$SEED" --sessions 20 --seed 16000001 > "$OUT/evaluation-s$SEED.log" 2>&1
done
echo summarizing > "$OUT/status.txt"
"$PY" scripts/summarize-liam-comparison.py "$OUT"
echo complete > "$OUT/status.txt"
