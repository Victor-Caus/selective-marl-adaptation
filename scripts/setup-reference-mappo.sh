#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_ROOT="$PROJECT_ROOT/.external"
SOURCE_ROOT="$EXTERNAL_ROOT/reference-sources/marlbenchmark-on-policy"
ENV_PREFIX="$EXTERNAL_ROOT/envs/mappo-reference-linux"
LOCK_FILE="$PROJECT_ROOT/reproductions/references.lock.json"

read -r SOURCE_URL SOURCE_COMMIT < <(
  python3 - "$LOCK_FILE" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
source = next(item for item in lock["sources"] if item["id"] == "marlbenchmark-on-policy")
print(source["repository"], source["commit"])
PY
)

mkdir -p "$(dirname "$SOURCE_ROOT")" "$(dirname "$ENV_PREFIX")"
if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$SOURCE_ROOT"
fi
git -C "$SOURCE_ROOT" fetch origin "$SOURCE_COMMIT" --depth 1
git -C "$SOURCE_ROOT" checkout --detach "$SOURCE_COMMIT"
ACTUAL_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$SOURCE_COMMIT" ]]; then
  echo "Reference commit mismatch: expected $SOURCE_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    python=3.8 pip=23.0 setuptools=65.6.3 wheel=0.38.4
fi

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install \
  numpy==1.23.5 \
  scipy==1.10.1 \
  torch==1.13.1 \
  gym==0.17.2 \
  tensorboardX==2.6.2.2 \
  protobuf==3.20.3 \
  setproctitle==1.3.3 \
  imageio==2.9.0 \
  wandb==0.15.12 \
  absl-py==0.9.0 \
  seaborn==0.10.1 \
  matplotlib==3.5.3
"$PYTHON" -m pip install --no-deps -e "$SOURCE_ROOT"

"$PYTHON" - <<'PY'
import gym
import numpy
import onpolicy
import scipy
import tensorboardX
import torch

print("python environment ready")
print("gym", gym.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("tensorboardX", tensorboardX.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("onpolicy", onpolicy.__file__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the Linux reference profile on this host")
PY

echo "Official MAPPO Linux environment ready at $ENV_PREFIX"
