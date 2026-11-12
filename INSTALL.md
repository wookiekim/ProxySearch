# Environment setup

ProxySearch uses one main environment and two optional auxiliary venvs. The
auxiliary venvs exist because ImageReward and the upstream (mmdet) GenEval
scorer pin `transformers` 4.x, while FLUX and SD3.5 in `diffusers` need
`transformers` 5.x.

| Environment | Used for | Built by |
|---|---|---|
| main (Python 3.12) | generation, search, CLIP / aesthetic / PickScore / HPSv2 verifiers, native GenEval scorer | `pip install -r requirements.txt` |
| ImageReward venv (Python 3.10) | `image_reward` verifier | `scripts/setup/build_venv_ir.sh` |
| GenEval venv (Python 3.10) | upstream mmdet GenEval scorer (`scripts/score_geneval.py`) | `scripts/setup/build_venv_geneval.sh` |

## Main environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login        # accept the FLUX.1-dev and SD3.5 licenses on Hugging Face first
```

We ran the paper experiments with torch 2.10 (CUDA 12.8) on A100-80GB GPUs.
If the default `torch` wheel does not match your CUDA driver, install torch
from https://pytorch.org before the requirements. Model weights download to the
standard Hugging Face cache; set `HF_HOME` to put it elsewhere.

`run_joint_geneval_lf.py` scores GenEval in-process by default
(`--scorer native`, Hugging Face Mask2Former + CLIP), so the main environment
is enough for the quick-start commands in the README.

## ImageReward venv (optional)

Needed only if `image_reward` is in your verifier list.

```bash
export PROXYSEARCH_IR_VENV=$PWD/.venv_ir          # default: /opt/venv_ir
bash scripts/setup/build_venv_ir.sh               # requires uv
export PROXYSEARCH_IR_PYTHON=$PROXYSEARCH_IR_VENV/bin/python
```

`VerifierEnsemble` calls `scripts/score_image_reward.py` with
`$PROXYSEARCH_IR_PYTHON` in a subprocess.

## GenEval venv (optional)

Needed only for the upstream mmdet GenEval scorer.

```bash
export PROXYSEARCH_GENEVAL_VENV=$PWD/.venv_geneval   # default: /opt/venv_geneval
bash scripts/setup/build_venv_geneval.sh             # requires uv; installs torch 2.0.1+cu118, mmcv-full 1.7.2, mmdet 2.28.2
bash benchmarks/geneval/evaluation/download_models.sh benchmarks/geneval/models
export PROXYSEARCH_GENEVAL_PYTHON=$PROXYSEARCH_GENEVAL_VENV/bin/python
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PROXYSEARCH_IR_PYTHON` | `/opt/venv_ir/bin/python` | python of the ImageReward venv |
| `PROXYSEARCH_IR_VENV` | `/opt/venv_ir` | where `build_venv_ir.sh` creates the venv |
| `PROXYSEARCH_GENEVAL_VENV` | `/opt/venv_geneval` | where `build_venv_geneval.sh` creates the venv |
| `PROXYSEARCH_GENEVAL_PYTHON` | (none) | python of the GenEval venv, for `scripts/score_geneval.py` |
| `PYTHONHASHSEED` | (unset) | set to `0` so noise IDs match across processes |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face model cache |

## Tests

```bash
pytest tests
python scripts/run_proxy_search.py --dry-run
```
