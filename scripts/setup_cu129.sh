#!/usr/bin/env bash
# Compatibility profile for the observed remote R550 driver. Validate before training.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/python"
UV="$PROJECT_ROOT/bootstrap/bin/uv"
mkdir -p "$PROJECT_ROOT/tmp"
export TMPDIR="$PROJECT_ROOT/tmp"
if [[ ! -x "$UV" ]]; then
  python3 -m pip install --target "$PROJECT_ROOT/bootstrap" uv
fi
"$UV" venv --python 3.12 --allow-existing .venv
"$UV" pip install --python .venv/bin/python \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
WHEEL_ARGS=()
if [[ -d "$PROJECT_ROOT/wheelhouse" ]]; then
  WHEEL_ARGS=(--find-links "$PROJECT_ROOT/wheelhouse")
fi
"$UV" pip install --python .venv/bin/python "${WHEEL_ARGS[@]}" \
  'https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
if [[ ! -d vendor/verl/.git ]]; then
  git clone --depth 1 --branch v0.9.1 https://github.com/verl-project/verl.git vendor/verl
fi
[[ "$(git -C vendor/verl rev-parse HEAD)" == 1876b06d0a3e4e71e06230be10af14492ca8a75b ]]
# Install core deps only; upstream fsdp/vllm extras would force CUDA 13 wheels.
"$UV" pip install --python .venv/bin/python "${WHEEL_ARGS[@]}" -e 'vendor/verl[verl-core]' matplotlib \
  'ray[default]==2.55.1' 'grpcio==1.81.1' 'protobuf==6.33.6' 'transformers==5.9.0'
"$UV" pip install --python .venv/bin/python --no-deps -e .
"$UV" pip install --python .venv/bin/python --no-deps fla-core==0.6.0 flash-linear-attention==0.6.0
python3 -c 'import json; from pathlib import Path; Path("environment.kernels.json").write_text(json.dumps({"tag":"fla060","fla-core":"0.6.0","flash-linear-attention":"0.6.0"},indent=2)+"\n")'
"$UV" pip check --python .venv/bin/python
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.device_count()==2; a=torch.randn(256,256,device="cuda"); print((a@a).sum().item()); import vllm; print(vllm.__version__)'
"$UV" pip freeze --python .venv/bin/python > environment.cu129.freeze.txt
