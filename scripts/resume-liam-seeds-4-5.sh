#!/usr/bin/env bash
# User-authorized recovery, 2026-09-29. Same frozen training/evaluation protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=.external/envs/selective-marl-linux/bin/python
OUT=results/liam-comparison-20260928-r1
ARCHIVE="$OUT/interrupted-seed4-20260929"
test ! -e "$ARCHIVE"
test ! -e "$OUT/train-s5"
test ! -e "$OUT/evaluation-s4"
test ! -e "$OUT/REPORT.md"
for SEED in 1 2 3; do
  test -f "$OUT/evaluation-s$SEED/COMPLETE"
  "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"]=="complete" and s["completed_steps"]==40000000' "$OUT/train-s$SEED/status.json"
done
"$PY" -m pytest -q > "$OUT/recovery-audit-tests.log" 2>&1
mkdir "$ARCHIVE"
mv "$OUT/train-s4" "$OUT/train-s4.log" "$ARCHIVE/"
cp "$OUT/status.txt" "$ARCHIVE/campaign-status-before.txt"
printf '%s\n' 'Interrupted at 24800800 steps; no optimizer/normalizer/RNG checkpoint. Restart same seed from scratch; do not use partial run for selection.' > "$ARCHIVE/RECOVERY.txt"
trap 'printf "failed recovery line=%s\n" "$LINENO" > "$OUT/status.txt"' ERR
set -x
for SEED in 4 5; do
  BASE="results/selectivity-controls-20260928-r1/base-s$SEED"
  MEMORY="results/selectivity-controls-20260928-r1/memory-s$SEED/memory.pt"
  sha256sum "$BASE/models/actor.pt" "$MEMORY" >> "$OUT/recovery-partner-memory-sha256.txt"
  echo "liam_training seed=$SEED recovery=20260929" > "$OUT/status.txt"
  "$PY" -u scripts/train-liam-reference.py --actor "$BASE/models/actor.pt" --output "$OUT/train-s$SEED" --seed "$SEED" --steps 40000000 --envs 32 --matched-steps 3420800 > "$OUT/train-s$SEED.log" 2>&1
  sha256sum "$OUT/train-s$SEED/matched.pt" "$OUT/train-s$SEED/final.pt" >> "$OUT/liam-checkpoints-sha256.txt"
  echo "evaluation seed=$SEED" > "$OUT/status.txt"
  "$PY" -u scripts/evaluate-liam-comparison.py --actor "$BASE/models/actor.pt" --memory "$MEMORY" --matched "$OUT/train-s$SEED/matched.pt" --long "$OUT/train-s$SEED/final.pt" --output "$OUT/evaluation-s$SEED" --sessions 20 --seed 16000001 > "$OUT/evaluation-s$SEED.log" 2>&1
done
echo summarizing > "$OUT/status.txt"
"$PY" scripts/summarize-liam-comparison.py "$OUT"
echo complete > "$OUT/status.txt"
