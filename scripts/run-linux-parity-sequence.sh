#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PYTHON="$PROJECT_ROOT/.external/envs/selective-marl-linux/bin/python"
REFERENCE_PYTHON="$PROJECT_ROOT/.external/envs/mappo-reference-linux/bin/python"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="$PROJECT_ROOT/results/runs"
REFERENCE_INTERVENTION_ROOT="$PROJECT_ROOT/results/reference-interventions/$TIMESTAMP"
mkdir -p "$OUTPUT_ROOT" "$REFERENCE_INTERVENTION_ROOT"

if [[ ! -x "$PROJECT_PYTHON" || ! -x "$REFERENCE_PYTHON" ]]; then
  echo "Run both Linux setup scripts before starting the sequence" >&2
  exit 1
fi

echo "stage=ours_no_interventions started_at=$(date -u +%FT%TZ)"
"$PROJECT_PYTHON" -u -m selective_marl pipeline \
  --profile research \
  --algorithms rmappo \
  --seeds 42 \
  --training-conditions none \
  --device cuda \
  --output "$OUTPUT_ROOT"
echo "stage=ours_no_interventions completed_at=$(date -u +%FT%TZ)"

MODEL_DIR="$({
  find "$PROJECT_ROOT"/results/reference-runs/linux_full_* \
    -type f -name actor.pt -printf '%T@ %h\n' 2>/dev/null || true
} | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "$MODEL_DIR" || ! -f "$MODEL_DIR/actor.pt" ]]; then
  echo "No completed official Linux full-run actor was found" >&2
  exit 1
fi

echo "stage=official_interventions started_at=$(date -u +%FT%TZ) model=$MODEL_DIR"
PYTHONPATH="$PROJECT_ROOT/src" "$REFERENCE_PYTHON" -u \
  "$PROJECT_ROOT/scripts/evaluate-reference-interventions.py" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$REFERENCE_INTERVENTION_ROOT" \
  --sessions 10 \
  --episodes 40 \
  --change-episode 20 \
  --seed 1001
echo "stage=official_interventions completed_at=$(date -u +%FT%TZ)"

"$PROJECT_PYTHON" -m pip freeze > "$REFERENCE_INTERVENTION_ROOT/project-environment.txt"
"$REFERENCE_PYTHON" -m pip freeze > "$REFERENCE_INTERVENTION_ROOT/reference-environment.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$REFERENCE_INTERVENTION_ROOT/project-commit.txt"
echo "Sequence complete"
