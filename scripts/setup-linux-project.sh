#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$PROJECT_ROOT/.external/envs/selective-marl-linux"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 1
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
    python=3.11 pip=26.0
fi

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install \
  torch==2.14.0 \
  mpe2==1.1.1 \
  numpy==2.4.6 \
  Pillow==12.3.0 \
  PyYAML==6.0.3 \
  matplotlib==3.11.2 \
  pandas==3.0.6 \
  seaborn==0.13.2 \
  pytest==9.1.1 \
  ruff==0.16.9
"$PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"

"$PYTHON" -m selective_marl doctor
"$PYTHON" -m pytest -q "$PROJECT_ROOT/tests"
echo "Project Linux environment ready at $ENV_PREFIX"
