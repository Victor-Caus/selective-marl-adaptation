#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-smoke}"
SEED="${2:-1}"
case "$PROFILE" in
  smoke) NUM_ENV_STEPS=32000 ;;
  pilot) NUM_ENV_STEPS=320000 ;;
  full) NUM_ENV_STEPS=3000000 ;;
  *) echo "Usage: $0 {smoke|pilot|full} [seed]" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$PROJECT_ROOT/.external/reference-sources/marlbenchmark-on-policy"
ENV_PREFIX="$PROJECT_ROOT/.external/envs/mappo-reference-linux"
PYTHON="$ENV_PREFIX/bin/python"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXPERIMENT="linux_${PROFILE}_s${SEED}_${TIMESTAMP}"
RUN_ROOT="$PROJECT_ROOT/results/reference-runs/$EXPERIMENT"
OFFICIAL_SCRIPTS="$SOURCE_ROOT/onpolicy/scripts"
OFFICIAL_OUTPUT="$OFFICIAL_SCRIPTS/results/MPE/simple_reference/rmappo/$EXPERIMENT"
SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
mkdir -p "$RUN_ROOT"

COMMAND=(
  "$PYTHON" -u train/train_mpe.py
  --env_name MPE
  --algorithm_name rmappo
  --experiment_name "$EXPERIMENT"
  --scenario_name simple_reference
  --num_agents 2
  --num_landmarks 3
  --seed "$SEED"
  --n_training_threads 1
  --n_rollout_threads 128
  --num_mini_batch 1
  --episode_length 25
  --num_env_steps "$NUM_ENV_STEPS"
  --ppo_epoch 15
  --gain 0.01
  --lr 0.0007
  --critic_lr 0.0007
  --use_wandb
)

"$PYTHON" - "$RUN_ROOT/metadata.json" "$PROFILE" "$SEED" "$NUM_ENV_STEPS" \
  "$SOURCE_COMMIT" "$EXPERIMENT" <<'PY'
import datetime
import json
import sys

path, profile, seed, steps, commit, experiment = sys.argv[1:]
metadata = {
    "status": "running",
    "profile": profile,
    "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_repository": "https://github.com/marlbenchmark/on-policy",
    "source_commit": commit,
    "algorithm": "rmappo",
    "scenario": "simple_reference",
    "experiment": experiment,
    "seed": int(seed),
    "num_env_steps": int(steps),
    "n_rollout_threads": 128,
    "execution": "official Linux SubprocVecEnv",
}
json.dump(metadata, open(path, "w", encoding="utf-8"), indent=2)
PY

"$PYTHON" -m pip freeze > "$RUN_ROOT/environment.txt"
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/command.txt"
printf '\n' >> "$RUN_ROOT/command.txt"

set +e
(
  cd "$OFFICIAL_SCRIPTS"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${COMMAND[@]}"
) 2>&1 | tee "$RUN_ROOT/console.log"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e

if [[ -d "$OFFICIAL_OUTPUT" ]]; then
  cp -a "$OFFICIAL_OUTPUT" "$RUN_ROOT/official-output"
fi

if [[ "$TRAIN_EXIT" -eq 0 ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/analyze-reference-mappo.py" --run-dir "$RUN_ROOT"
  MODEL_DIR="$(find "$RUN_ROOT/official-output" -type f -name actor.pt -printf '%h\n' | head -n 1)"
  EVALUATION_ARGS=(
    "$PROJECT_ROOT/scripts/evaluate-reference-mappo.py"
    --model-dir "$MODEL_DIR"
    --output-dir "$RUN_ROOT/evaluation"
    --episodes 200
    --seed 1001
  )
  if command -v xvfb-run >/dev/null 2>&1; then
    MPLBACKEND=Agg xvfb-run -a "$PYTHON" "${EVALUATION_ARGS[@]}" --save-gif
  else
    MPLBACKEND=Agg "$PYTHON" "${EVALUATION_ARGS[@]}"
  fi
fi

"$PYTHON" - "$RUN_ROOT/metadata.json" "$TRAIN_EXIT" <<'PY'
import datetime
import json
import sys

path, exit_code = sys.argv[1], int(sys.argv[2])
metadata = json.load(open(path, encoding="utf-8"))
metadata["status"] = "complete" if exit_code == 0 else "failed"
metadata["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
metadata["exit_code"] = exit_code
json.dump(metadata, open(path, "w", encoding="utf-8"), indent=2)
PY

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "Reference training failed with exit code $TRAIN_EXIT" >&2
  exit "$TRAIN_EXIT"
fi
echo "Reference training and evaluation complete at $RUN_ROOT"
