#!/usr/bin/env bash
# Build the ImageReward venv at $PROXYSEARCH_IR_VENV (default /opt/venv_ir).
# ImageReward requires transformers 4.x which conflicts with main venv's 5.x.
set -euo pipefail

VENV="${PROXYSEARCH_IR_VENV:-/opt/venv_ir}"
LOG="${VENV}.build.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== build venv_ir start $(date) ==="

if [ -d $VENV ]; then rm -rf $VENV; fi
uv venv --python "$(uv python find 3.10)" $VENV
PY=$VENV/bin/python
PIP="uv pip install --python $PY"

# torch pinned (match mmcv/mmdet torch for tooling compat; torch 2.0.1 works fine for IR)
$PIP "setuptools<81" "wheel" "numpy<2.0"
$PIP "torch==2.0.1" "torchvision==0.15.2" \
  --index-url https://download.pytorch.org/whl/cu118

# transformers pin for ImageReward
$PIP "transformers==4.36.2" "tokenizers" "safetensors" "huggingface-hub"

# ImageReward + deps
$PIP "image-reward" "ftfy" "regex" "Pillow" "tqdm" "requests" "timm"
$PIP "fairscale"

echo "=== verifying ==="
$PY -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import transformers; print('transformers', transformers.__version__)
import ImageReward
print('ImageReward import OK')
print('ALL PASS')
"

echo "=== venv_ir DONE $(date) ==="
