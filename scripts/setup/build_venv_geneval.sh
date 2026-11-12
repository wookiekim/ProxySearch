#!/usr/bin/env bash
# Build the GenEval (mmdet) venv at $PROXYSEARCH_GENEVAL_VENV (default /opt/venv_geneval).
# Verifies the torch version after each step: mmdet and other deps can silently
# downgrade torch after the mmcv wheel (built for torch 2.0) is installed.
set -euo pipefail

VENV="${PROXYSEARCH_GENEVAL_VENV:-/opt/venv_geneval}"
LOG="${VENV}.build.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== build start $(date) ==="

if [ -d $VENV ]; then rm -rf $VENV; fi
uv venv --python "$(uv python find 3.10)" $VENV
source $VENV/bin/activate
PIP="uv pip install --python $VENV/bin/python"
PY=$VENV/bin/python

verify_torch() {
  $PY -c "
import torch
v = torch.__version__
assert v.startswith('2.0.1'), f'torch version drift: expected 2.0.1, got {v}'
assert '+cu118' in v, f'cuda variant drift: expected cu118, got {v}'
print('  [OK] torch:', v, '| cuda:', torch.version.cuda)
"
}

echo "--- 1/6 base tooling ---"
$PIP "setuptools<81" "wheel" "numpy<2.0"

echo "--- 2/6 torch 2.0.1+cu118 (pin) ---"
$PIP "torch==2.0.1" "torchvision==0.15.2" \
  --index-url https://download.pytorch.org/whl/cu118
verify_torch

echo "--- 3/6 mmcv-full 1.7.2 precompiled wheel for cu118/torch2.0 ---"
$PIP "mmcv-full==1.7.2" \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
  --no-deps
verify_torch

echo "--- 3.5/6 runtime deps BEFORE mmcv import test (dep-resolved, torch-guarded) ---"
$PIP \
  "matplotlib" "pycocotools" "six" "terminaltables" \
  "addict" "yapf==0.40.1" "packaging" "pyyaml" \
  "shapely" "scipy" "tqdm" "pandas" "tifffile" \
  "opencv-python-headless" "platformdirs"
verify_torch

$PY -c "
import mmcv
print('  [mmcv]', mmcv.__version__)
from mmcv.ops import RoIPool
print('  [mmcv.ops OK]')
"

echo "--- 4/6 mmdet 2.28.2 --no-deps ---"
$PIP --no-deps "mmdet==2.28.2"
verify_torch

echo "--- 6/6 CLIP + benchmark ---"
$PIP --no-deps \
  "open-clip-torch==2.20.0" "clip-benchmark==1.4.0" \
  "ftfy" "regex" "sentencepiece" "protobuf" "wcwidth" \
  "huggingface-hub" "timm" "safetensors"
verify_torch

echo "=== VERIFY ALL ==="
$PY -c "
import torch, mmcv, mmdet
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
print('mmcv', mmcv.__version__)
print('mmdet', mmdet.__version__)
from mmcv.ops import RoIPool
print('mmcv.ops OK')
from mmdet.apis import init_detector, inference_detector
print('mmdet.apis OK')
import open_clip
print('open_clip OK')
print('ALL PASS')
"

echo "=== DONE $(date) ==="
