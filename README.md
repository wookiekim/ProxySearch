# ProxySearch

**ProxySearch: Decoupled Inference-Time Scaling for Diffusion Models via Asymmetric Noise-Rank Transfer**

Seungwook Kim<sup>1</sup>, Jongmin Lee<sup>2</sup>
<sup>1</sup>GenGenAI &nbsp; <sup>2</sup>Chung-Ang University

**NeurIPS 2026** · [Project page](https://wookiekim.github.io/ProxySearch/)

ProxySearch is a training-free inference-time scaling method for text-to-image
diffusion models. Closed-form image statistics (brightness, contrast,
saturation, ...) preserve the ranking of initial noises from a distilled
few-step proxy to its multi-step teacher, while learned semantic scorers do
not. ProxySearch uses this asymmetry: a 1-step proxy filters noise candidates
by statistical match, and the teacher then picks the best survivor under a
semantic reward of your choice.

Supported proxy → teacher pairs: FLUX.1-schnell → FLUX.1-dev,
SD3.5-Large-Turbo → SD3.5-Large, SDXL-Turbo → SDXL, DMD2 → SDXL.

## Installation

```bash
git clone https://github.com/wookiekim/ProxySearch.git
cd ProxySearch
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt                     # install a torch build matching your CUDA first if needed
hf auth login                                       # FLUX.1-dev and SD3.5 weights are gated on Hugging Face
```

The ImageReward verifier and the upstream GenEval scorer need older
`transformers` and run in their own venvs. See [INSTALL.md](INSTALL.md).

## Quick start

```bash
export PYTHONHASHSEED=0   # keeps noise IDs stable across runs

# Sanity check, no GPU or weights needed
python scripts/run_proxy_search.py --dry-run

# Proxy -> teacher rank transfer: per-verifier Spearman rho over 32 noises per prompt
python scripts/run_rank_correlation.py --config configs/flux_schnell_dev.yaml \
    --num-prompts 100 --num-noises 32 --image-size 512 \
    --verifiers brightness,color_temperature,compressibility,saturation,sharpness,contrast,colorfulness,edge_density,clip,aesthetic \
    --output-dir outputs/correlation/flux

# ProxySearch (m3) vs. teacher-only argmax (m1) on GenEval with a statistical target
python scripts/run_joint_geneval_lf.py --model-family flux \
    --num-prompts 50 --num-noises-proxy 64 --num-noises-teacher 8 --top-k 8 \
    --lf-targets bright_high_contrast,dark_moody --methods m1,m3 \
    --proxy-steps 1 --teacher-steps 50 --image-size 1024 \
    --output-dir outputs/joint_geneval_lf/flux
```

Use `--model-family sd3` or `sdxl` (and the matching `configs/*.yaml`) for the
other pairs, and `--hf-objective pickscore` or `hpsv2` to swap the semantic
reward. Run any script with `--help` for all options.

## Repository layout

```
src/            models (proxy/teacher/verifier loaders), noise sampling, search, analysis
scripts/        entry points and scorers; scripts/setup/ builds the auxiliary venvs
configs/        proxy/teacher pair configs
benchmarks/     GenEval prompts and evaluation code (MIT, from djghosh13/geneval)
tests/          unit tests (pytest)
```

## Citation

```bibtex
@inproceedings{kim2026proxysearch,
  title     = {ProxySearch: Decoupled Inference-Time Scaling for Diffusion Models via Asymmetric Noise-Rank Transfer},
  author    = {Kim, Seungwook and Lee, Jongmin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). `benchmarks/geneval/` is from
[GenEval](https://github.com/djghosh13/geneval) under its own MIT license.
