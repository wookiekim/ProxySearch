#!/usr/bin/env python3
"""Joint-constraint GenEval + LF evaluation.

Compares 6 methods on (prompt, LF target) pairs (main paper experiment).

Methods:
  M0    Random            : first of K random teacher gens (control)
  M1    Teacher-GenEval   : argmax GenEval over N_t teacher gens
  M1p5  Teacher-scaled    : argmax GenEval over N_t+K teacher gens (compute-matched to M3)
  M2    Teacher-composite : argmax (GenEval + lambda * (-LF_L1)) — sweeps lambda
  M3    Decoupled (ours)  : proxy x N_p → top-K by LF → teacher x K → argmax GenEval
  M3alt Teacher+postproc  : M1 image + linear brightness/contrast shift to LF target

LF targets (globally fixed):
  bright_high_contrast : brightness=0.72, contrast=0.26
  dark_moody           : brightness=0.22, contrast=0.20
Target L1 uses brightness + contrast axes only (other 6 LF dims logged but not weighted).

Output: per-(prompt x target x method [x lambda]) JSONL row.

Protocol details:
  1. Global LF targets (not per-prompt)
  2. lambda sweep for M2 reported at argmax-GenEval
  3. Feasibility probe per prompt — flag infeasible target cells
  4. McNemar + bootstrap CI in summary
  5. Same pinned noise pool across methods (noise_from_id seed)

NoiseSampler dtype: fp16 default (no dtype kwarg) — bf16 noise breaks FLUX-schnell rank transfer.
PYTHONHASHSEED=0 required (export it before launching).
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline

from src.models.proxy_loader import ProxyLoader
from src.models.teacher_loader import TeacherLoader
from src.models.verifier_loader import VerifierEnsemble
from src.search.noise_sampler import NoiseSampler
from src.utils.seed import set_seed


# --- constants ---

LF_VERIFIERS = ["brightness", "compressibility", "contrast", "colorfulness",
                "saturation", "sharpness", "edge_density", "color_temperature"]

# Target L1 uses any axis present in target dict. Others get weight 0.
LF_TARGET_AXES = ["brightness", "compressibility", "contrast", "colorfulness",
                  "saturation", "sharpness", "edge_density", "color_temperature"]

LF_TARGETS: Dict[str, Dict[str, float]] = {
    "bright_high_contrast": {"brightness": 0.72, "contrast": 0.26},
    "dark_moody":           {"brightness": 0.22, "contrast": 0.20},
    # Generality targets (single-axis), values from empirical p75/p25 of the LF pool
    "sharp_high":           {"sharpness": 1000.0},
    "sharp_low":            {"sharpness": 200.0},
    "warm":                 {"color_temperature": 0.5},
    "cool":                 {"color_temperature": -0.1},
}

DEFAULT_LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0]

# GenEval JSONL uses 'color_attr' (not 'color_attribution'); verified in
# benchmarks/geneval/prompts/evaluation_metadata.jsonl tag counts.
GENEVAL_STRATA = ["single_object", "two_object", "counting", "colors",
                  "position", "color_attr"]


# --- helpers ---

def prepare_flux_latents(pipeline, latents):
    """FLUX needs packed latents."""
    if isinstance(pipeline, FluxPipeline):
        B, C, H, W = latents.shape
        return pipeline._pack_latents(latents, B, C, H, W)
    return latents


def flux_generate(pipeline, noise, prompt, num_steps, guidance, image_size):
    lat = prepare_flux_latents(pipeline, noise)
    with torch.no_grad():
        out = pipeline(prompt=prompt, latents=lat,
                       num_inference_steps=num_steps, guidance_scale=guidance,
                       height=image_size, width=image_size, output_type="pil")
    return out.images[0]


def score_lf(images: List[Image.Image], verifier: VerifierEnsemble) -> List[Dict[str, float]]:
    """Return [{verifier: score}] per image."""
    prompts = [""] * len(images)
    scored = verifier.score_individual(images, prompts)
    return [{k: float(scored[k][i]) for k in LF_VERIFIERS} for i in range(len(images))]


def lf_l1(lf_scores: Dict[str, float], target: Dict[str, float]) -> float:
    """L1 distance on target axes only."""
    return sum(abs(lf_scores[k] - v) for k, v in target.items())


def metadata_to_prompt_info(md: Dict, idx: int) -> Dict:
    """Convert GenEval JSONL row to GenEvalEvaluator.prompt_info format.

    Keeps `raw` (full metadata dict) for use with NativeGenEvalScorer which
    needs counting/color/position fields.
    """
    include = md.get("include", [])
    exclude = md.get("exclude", [])
    objects = [c["class"] for c in include]
    count = include[0]["count"] if include else 1
    return {
        "prompt_id": f"gen_{idx}",
        "prompt": md["prompt"],
        "category": md["tag"],
        "objects": objects,
        "exclude": [c["class"] for c in exclude],
        "count": count,
        "raw": md,
    }


def load_geneval_prompts(path: Path) -> List[Dict]:
    prompts = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            md = json.loads(line)
            prompts.append(metadata_to_prompt_info(md, i))
    return prompts


def stratified_sample(prompts: List[Dict], n: int, seed: int) -> List[Dict]:
    """Sample n prompts stratified across GENEVAL_STRATA."""
    rng = random.Random(seed)
    by_tag: Dict[str, List[Dict]] = {t: [] for t in GENEVAL_STRATA}
    for p in prompts:
        if p["category"] in by_tag:
            by_tag[p["category"]].append(p)
    per_tag = max(1, n // len(GENEVAL_STRATA))
    out = []
    for t in GENEVAL_STRATA:
        pool = by_tag.get(t, [])
        if not pool:
            continue
        k = min(per_tag, len(pool))
        out.extend(rng.sample(pool, k))
    # Top up if short
    remaining = [p for p in prompts if p not in out]
    while len(out) < n and remaining:
        out.append(rng.choice(remaining))
        remaining = [p for p in prompts if p not in out]
    return out[:n]


def postprocess_shift(img: Image.Image, cur_lf: Dict[str, float],
                      target: Dict[str, float]) -> Image.Image:
    """Linear brightness + contrast shift toward target. M3alt baseline."""
    from PIL import ImageEnhance
    arr = np.asarray(img).astype(np.float32) / 255.0
    cur_b = cur_lf.get("brightness", float(arr.mean()))
    tgt_b = target.get("brightness", cur_b)
    delta_b = tgt_b - cur_b
    arr = np.clip(arr + delta_b, 0, 1)

    out = Image.fromarray((arr * 255).astype(np.uint8))
    # Contrast adjustment via PIL (approximate)
    cur_c = cur_lf.get("contrast", 0.2)
    tgt_c = target.get("contrast", cur_c)
    if cur_c > 1e-4:
        factor = tgt_c / cur_c
        factor = float(np.clip(factor, 0.2, 3.0))
        out = ImageEnhance.Contrast(out).enhance(factor)
    return out


# --- GenEval scoring (in-process OwlViT fallback) ---

class GenEvalOwlScorer:
    """In-process GenEval scoring via OwlViT (proxy for real pipeline).

    Loads once, scores many. Returns per-image binary success + confidence.
    Covers {single_object, two_object, counting, colors, position, color_attribution}
    by using object-presence as a proxy (official pipeline uses Mask2Former for
    counting; OwlViT is the closest single-venv approximation).
    """
    def __init__(self, device: str = "cuda", threshold: float = 0.1):
        self.device = device
        self.threshold = threshold
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.detector = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32").to(device).eval()

    def detect(self, image: Image.Image, queries: List[str]) -> List[Dict]:
        inputs = self.processor(text=[queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.detector(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        # transformers>=4.50 renamed to post_process_grounded_object_detection
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes,
                threshold=self.threshold,
            )[0]
            # new API returns label names directly in "text_labels" / "labels"
            # be tolerant of both
        else:
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes,
                threshold=self.threshold,
            )[0]
        dets = []
        scores_t = results["scores"]
        boxes_t = results["boxes"]
        label_strs = results.get("text_labels")
        label_ids = results.get("labels")
        for idx, (s, b) in enumerate(zip(scores_t, boxes_t)):
            # Prefer text_labels if populated, else map label index to query
            label = None
            if label_strs is not None and idx < len(label_strs) and label_strs[idx] is not None:
                label = str(label_strs[idx])
            elif label_ids is not None:
                li = label_ids[idx]
                li_int = int(li.item()) if hasattr(li, "item") else int(li)
                if 0 <= li_int < len(queries):
                    label = queries[li_int]
            if label is None:
                continue
            dets.append({"label": label, "score": float(s),
                          "box": b.cpu().tolist()})
        return dets

    def score(self, image: Image.Image, prompt_info: Dict) -> Dict:
        """Returns {'success': bool, 'confidence': float, 'details': ...}."""
        cat = prompt_info["category"]
        objs = prompt_info.get("objects", [])
        exclude = prompt_info.get("exclude", [])
        count = prompt_info.get("count", 1)

        queries = list(objs) + list(exclude) if exclude else list(objs)
        if not queries:
            return {"success": False, "confidence": 0.0, "reason": "no_objects"}

        dets = self.detect(image, queries)
        label_scores: Dict[str, float] = {}
        label_counts: Dict[str, int] = {}
        for d in dets:
            if d["score"] >= self.threshold:
                label_scores[d["label"]] = max(label_scores.get(d["label"], 0), d["score"])
                label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1

        if cat == "two_object" and len(objs) >= 2:
            ok = all(o in label_scores for o in objs[:2])
            conf = float(np.mean([label_scores.get(o, 0.0) for o in objs[:2]]))
        elif cat == "counting" and objs:
            ok = label_counts.get(objs[0], 0) == count
            conf = label_scores.get(objs[0], 0.0)
        else:
            # single_object, colors, position, color_attribution: require obj present,
            # and exclude-classes NOT present
            ok = bool(objs) and objs[0] in label_scores
            if exclude:
                ok = ok and not any(e in label_scores for e in exclude)
            conf = label_scores.get(objs[0] if objs else "", 0.0)
        return {
            "success": bool(ok),
            "confidence": float(conf),
            "detected": {k: {"score": v, "count": label_counts.get(k, 0)}
                          for k, v in label_scores.items()},
            "category": cat,
        }

    def score_batch(self, images: List[Image.Image], prompt_info: Dict) -> List[Dict]:
        return [self.score(im, prompt_info) for im in images]


# --- core experiment ---

def run_pilot(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    probe_path = out_dir / "feasibility_probe.jsonl"
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Prompts
    all_prompts = load_geneval_prompts(Path(args.geneval_prompts))
    if args.prompt_pin:
        # Load pinned prompt_id list for reproducibility
        pin = json.loads(Path(args.prompt_pin).read_text())
        pinned_ids = [p["prompt_id"] for p in pin["prompts"]]
        id_to_prompt = {p["prompt_id"]: p for p in all_prompts}
        prompts = [id_to_prompt[pid] for pid in pinned_ids if pid in id_to_prompt]
        print(f"Loaded pinned {len(prompts)} prompts from {args.prompt_pin}")
    elif args.stratified:
        prompts = stratified_sample(all_prompts, args.num_prompts, args.seed)
    else:
        prompts = all_prompts[: args.num_prompts]
    # Sharding for parallel runs
    if args.prompt_end is not None:
        prompts = prompts[args.prompt_start:args.prompt_end]
    elif args.prompt_start:
        prompts = prompts[args.prompt_start:]
    print(f"Selected {len(prompts)} prompts (stratified={args.stratified}, "
          f"start={args.prompt_start}, end={args.prompt_end})")
    for p in prompts:
        print(f"  [{p['prompt_id']:>8s}] {p['category']:>18s}  {p['prompt']}")

    targets = {k: LF_TARGETS[k] for k in args.lf_targets.split(",")}
    lambdas = [float(x) for x in args.lambda_sweep.split(",")]
    methods = set(args.methods.split(","))
    print(f"Targets: {list(targets)}")
    print(f"Lambdas: {lambdas}")
    print(f"Methods: {methods}")

    # --- load once ---
    device = "cuda"
    print(f"\nLoading {args.model_family}-family proxy...")
    proxy_loader = ProxyLoader(model_family=args.model_family, dtype=torch.bfloat16)
    proxy_pipe = proxy_loader.load()
    H = args.image_size
    latent_shape = proxy_loader.get_latent_shape(image_size=H)
    # fp16 default (no dtype kwarg) — bf16 noise breaks FLUX-schnell rank transfer
    sampler = NoiseSampler(latent_shape=latent_shape, device=device)

    print("Loading LF verifier...")
    verifier = VerifierEnsemble(verifiers=LF_VERIFIERS)

    # [HF-SWAP 2026-04-29] Optional learned-preference v_HF scorer.
    # If --hf-objective is "pickscore" or "hpsv2", load a separate
    # VerifierEnsemble holding only that v_HF and use it as the
    # selection criterion in M1/M2/M3 instead of GenEval pass.
    hf_objective = args.hf_objective
    if hf_objective != "geneval":
        print(f"Loading HF preference verifier: {hf_objective}")
        hf_verifier = VerifierEnsemble(verifiers=[hf_objective], device=device)
    else:
        hf_verifier = None

    def _hf_pool_scores(images, prompt):
        """Return scalar v_HF score per image for the chosen objective."""
        if hf_verifier is None:
            return None
        prompts = [prompt] * len(images)
        scored = hf_verifier.score_individual(images, prompts)
        return np.array(scored[hf_objective], dtype=np.float64)

    print(f"Loading GenEval scorer: {args.scorer}")
    if args.scorer == "native":
        from score_geneval_native import NativeGenEvalScorer
        geneval_raw = NativeGenEvalScorer(device=device)

        class _NativeAdapter:
            """Wrap NativeGenEvalScorer to match OwlViT adapter signature
            (takes prompt_info dict, internally uses prompt_info['raw'] metadata)."""
            def __init__(self, s): self.s = s
            def score(self, image, prompt_info):
                md = prompt_info.get("raw", {
                    "tag": prompt_info.get("category", "single_object"),
                    "include": [{"class": o, "count": prompt_info.get("count", 1)}
                                  for o in prompt_info.get("objects", [])],
                    "exclude": [{"class": e, "count": 1}
                                  for e in prompt_info.get("exclude", [])],
                })
                return self.s.score(image, md)
            def score_batch(self, images, prompt_info):
                return [self.score(im, prompt_info) for im in images]
        geneval = _NativeAdapter(geneval_raw)
    elif args.scorer == "owlvit":
        geneval = GenEvalOwlScorer(device=device, threshold=args.geneval_threshold)
    else:
        raise ValueError(f"unknown scorer {args.scorer}")

    # Resume support
    done = set()
    if results_path.exists():
        for line in open(results_path):
            try:
                r = json.loads(line)
                done.add((r["prompt_id"], r["lf_target"], r["method"], r.get("lambda", -1.0)))
            except Exception:
                pass
    print(f"Resuming: {len(done)} rows already logged")

    # --- per-prompt loop ---
    t_all = time.time()
    for pi, pinfo in enumerate(prompts):
        prompt = pinfo["prompt"]
        prompt_id = pinfo["prompt_id"]
        print(f"\n[{pi+1}/{len(prompts)}] {prompt_id} | {pinfo['category']} | {prompt}")

        # ----- proxy pool -----
        N_p = args.num_noises_proxy
        proxy_noises = [sampler.noise_from_id(args.seed, prompt_id, i).to(proxy_pipe.device)
                         for i in range(N_p)]
        t0 = time.time()
        proxy_images = []
        for i, n in enumerate(proxy_noises):
            img = flux_generate(proxy_pipe, n, prompt,
                                 num_steps=args.proxy_steps,
                                 guidance=args.proxy_guidance,
                                 image_size=H)
            proxy_images.append(img)
        t_proxy = time.time() - t0
        proxy_lf = score_lf(proxy_images, verifier)
        print(f"  proxy: {N_p} gens in {t_proxy:.1f}s  | "
              f"brightness range [{min(s['brightness'] for s in proxy_lf):.3f},"
              f"{max(s['brightness'] for s in proxy_lf):.3f}]")

        # Feasibility probe: record empirical range for each axis
        probe = {
            "prompt_id": prompt_id, "prompt": prompt,
            "proxy_empirical_range": {k: [float(min(s[k] for s in proxy_lf)),
                                           float(max(s[k] for s in proxy_lf))]
                                       for k in LF_VERIFIERS},
        }
        for t_name, t_vals in targets.items():
            probe_t = {}
            for ax in LF_TARGET_AXES:
                lo, hi = probe["proxy_empirical_range"][ax]
                probe_t[ax] = {
                    "target": t_vals.get(ax),
                    "in_range": bool(lo <= t_vals.get(ax, 0) <= hi),
                }
            probe[f"target_{t_name}"] = probe_t
        with open(probe_path, "a") as f:
            f.write(json.dumps(probe) + "\n")

    # ----- shared teacher pool (target-independent) -----
    # Only need to do this once per prompt; placed after proxy above
        # ====== LOAD TEACHER (only once per run, unload proxy first if memory) ======
        # Note: for A100 80GB, both FLUX-schnell + FLUX-dev fit; skip swap.
        # If OOM occurs, unload proxy here. For now, keep both loaded.
        if not hasattr(run_pilot, "_teacher"):
            print(f"  Loading {args.model_family}-family teacher (one-time)...")
            teacher_loader = TeacherLoader(model_family=args.model_family, dtype=torch.bfloat16)
            run_pilot._teacher = teacher_loader.load()
            run_pilot._teacher_loader = teacher_loader
        teacher_pipe = run_pilot._teacher

        N_t = args.num_noises_teacher
        K = args.top_k
        Nt_plus_K = N_t + K
        # Teacher pool only needs N_t+K images when m1p5 (argmax over N_t+K) is run;
        # otherwise N_t suffices. Cap at N_p (can't index beyond the proxy pool).
        pool_size = Nt_plus_K if "m1p5" in methods else N_t
        pool_size = min(pool_size, N_p)
        t0 = time.time()
        teacher_pool_images = []
        for i in range(pool_size):
            img = flux_generate(teacher_pipe,
                                 proxy_noises[i],  # same physical noise as proxy pool
                                 prompt,
                                 num_steps=args.teacher_steps,
                                 guidance=args.teacher_guidance,
                                 image_size=H)
            teacher_pool_images.append(img)
        t_teacher_shared = time.time() - t0
        teacher_pool_lf = score_lf(teacher_pool_images, verifier)
        print(f"  teacher pool: {Nt_plus_K} gens in {t_teacher_shared:.1f}s")

        # GenEval on teacher pool (always logged, regardless of selection criterion)
        teacher_pool_geneval = geneval.score_batch(teacher_pool_images, pinfo)

        # [HF-SWAP 2026-04-29] If --hf-objective != geneval, score the teacher
        # pool with the chosen learned-preference v_HF. None otherwise.
        teacher_pool_hf = (_hf_pool_scores(teacher_pool_images, prompt)
                            if hf_verifier is not None else None)

        # Save teacher pool images + meta for m_tseq vhf rescoring later.
        teacher_pool_paths = []
        if "m_tseq" in methods:
            for i, img in enumerate(teacher_pool_images[:N_t]):
                p = images_dir / f"{prompt_id}__teacher_pool__noise{i:02d}.png"
                if not p.exists():
                    img.save(p)
                teacher_pool_paths.append(str(p))
            tp_meta_path = out_dir / "teacher_pool_meta.jsonl"
            tp_meta_done = False
            if tp_meta_path.exists():
                for line in open(tp_meta_path):
                    try:
                        if json.loads(line).get("prompt_id") == prompt_id:
                            tp_meta_done = True
                            break
                    except Exception:
                        pass
            if not tp_meta_done:
                with open(tp_meta_path, "a") as f:
                    f.write(json.dumps({
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "category": pinfo["category"],
                        "objects": pinfo.get("objects", []),
                        "base_seed": args.seed,
                        "teacher_pool_size": N_t,
                        "teacher_pool_image_paths": teacher_pool_paths,
                        "teacher_pool_lf": teacher_pool_lf[:N_t],
                        "teacher_pool_geneval": [
                            {"success": bool(g["success"]),
                             "confidence": float(g.get("confidence", 0.0)),
                             "detected": g.get("detected", {})}
                            for g in teacher_pool_geneval[:N_t]
                        ],
                    }) + "\n")

        # ----- per-target methods -----
        for t_name, t_vals in targets.items():
            target_axes = {k: t_vals[k] for k in LF_TARGET_AXES if k in t_vals}

            # helper: LF L1 over a list
            def l1_list(lfs):
                return [lf_l1({k: v for k, v in lf.items()}, target_axes) for lf in lfs]

            # --- M0 Random (first 1 of teacher pool) ---
            if "m0" in methods and (prompt_id, t_name, "m0", -1.0) not in done:
                idx = 0  # deterministic "random" — first noise index
                img = teacher_pool_images[idx]
                lf = teacher_pool_lf[idx]
                gv = teacher_pool_geneval[idx]
                row = _row(prompt_id, pinfo, t_name, t_vals, "m0", None, idx,
                            args.seed, img, lf, gv,
                            target_axes, images_dir,
                            nfe=1 * args.teacher_steps,
                            wall_clock_s=t_teacher_shared / Nt_plus_K)
                _write_row(results_path, row)

            # --- M1 Teacher-v_HF (argmax over first N_t) ---
            # [HF-SWAP 2026-04-29] If hf_objective != geneval, select by
            # learned-preference scalar instead of GenEval pass.
            if "m1" in methods and (prompt_id, t_name, "m1", -1.0) not in done:
                sub_ge = teacher_pool_geneval[:N_t]
                if teacher_pool_hf is not None:
                    best_idx = _argmax_hf(teacher_pool_hf[:N_t],
                                            teacher_pool_lf[:N_t], target_axes)
                else:
                    best_idx = _argmax_ge(sub_ge, teacher_pool_lf[:N_t], target_axes,
                                           tiebreaker="lf_l1_min")
                img = teacher_pool_images[best_idx]
                lf = teacher_pool_lf[best_idx]
                gv = teacher_pool_geneval[best_idx]
                row = _row(prompt_id, pinfo, t_name, t_vals, "m1", None, best_idx,
                            args.seed, img, lf, gv, target_axes, images_dir,
                            nfe=N_t * args.teacher_steps,
                            wall_clock_s=(t_teacher_shared / Nt_plus_K) * N_t)
                _write_row(results_path, row)

                # M3alt: postprocess M1 image
                if "m3alt" in methods and (prompt_id, t_name, "m3alt", -1.0) not in done:
                    shifted = postprocess_shift(img, lf, target_axes)
                    # Rescore on shifted
                    shifted_lf = score_lf([shifted], verifier)[0]
                    shifted_gv = geneval.score(shifted, pinfo)
                    row = _row(prompt_id, pinfo, t_name, t_vals, "m3alt", None, best_idx,
                                args.seed, shifted, shifted_lf, shifted_gv,
                                target_axes, images_dir,
                                nfe=N_t * args.teacher_steps,  # same as M1, no extra NFE
                                wall_clock_s=(t_teacher_shared / Nt_plus_K) * N_t,
                                postprocess=True)
                    _write_row(results_path, row)

            # --- M1p5 Teacher-scaled (argmax over full Nt_plus_K) ---
            if "m1p5" in methods and (prompt_id, t_name, "m1p5", -1.0) not in done:
                if teacher_pool_hf is not None:
                    best_idx = _argmax_hf(teacher_pool_hf,
                                            teacher_pool_lf, target_axes)
                else:
                    best_idx = _argmax_ge(teacher_pool_geneval, teacher_pool_lf,
                                           target_axes, tiebreaker="lf_l1_min")
                img = teacher_pool_images[best_idx]
                lf = teacher_pool_lf[best_idx]
                gv = teacher_pool_geneval[best_idx]
                row = _row(prompt_id, pinfo, t_name, t_vals, "m1p5", None, best_idx,
                            args.seed, img, lf, gv, target_axes, images_dir,
                            nfe=Nt_plus_K * args.teacher_steps,
                            wall_clock_s=t_teacher_shared)
                _write_row(results_path, row)

            # --- M2 Composite (sweep lambda) ---
            # [HF-SWAP 2026-04-29] When hf_objective != geneval, the composite
            # is v_HF (continuous) + lam * (-LF_L1). When hf_objective ==
            # geneval, it's the original GenEval-pass-binary + lam * (-LF_L1).
            if "m2" in methods:
                for lam in lambdas:
                    if (prompt_id, t_name, "m2", lam) in done:
                        continue
                    sub_lf = teacher_pool_lf[:N_t]
                    sub_ge = teacher_pool_geneval[:N_t]
                    l1s = np.array(l1_list(sub_lf))
                    if teacher_pool_hf is not None:
                        sub_hf = teacher_pool_hf[:N_t]
                        comp = sub_hf + lam * (-l1s)
                        # tiebreak: smallest L1
                        order = np.lexsort((l1s, -comp))
                    else:
                        ge_scalar = np.array([1.0 if r["success"] else 0.0 for r in sub_ge])
                        comp = ge_scalar + lam * (-l1s)
                        # Primary tiebreaker: GenEval confidence, then -L1
                        tb = np.array([r.get("confidence", 0.0) for r in sub_ge])
                        # stable argmax: comp desc, then tb desc, then -l1 desc
                        order = np.lexsort((-l1s, -tb, -comp))
                    best_idx = int(order[0])
                    img = teacher_pool_images[best_idx]
                    lf = teacher_pool_lf[best_idx]
                    gv = teacher_pool_geneval[best_idx]
                    row = _row(prompt_id, pinfo, t_name, t_vals, "m2", lam, best_idx,
                                args.seed, img, lf, gv, target_axes, images_dir,
                                nfe=N_t * args.teacher_steps,
                                wall_clock_s=(t_teacher_shared / Nt_plus_K) * N_t)
                    _write_row(results_path, row)

            # --- M3 Decoupled (proxy top-K by LF → teacher × K → argmax GenEval) ---
            if "m3" in methods and (prompt_id, t_name, "m3", -1.0) not in done:
                proxy_l1 = np.array(l1_list(proxy_lf))
                topk = np.argsort(proxy_l1)[:K]
                t1 = time.time()
                m3_teacher_images = []
                for idx in topk:
                    img = flux_generate(teacher_pipe, proxy_noises[int(idx)],
                                         prompt,
                                         num_steps=args.teacher_steps,
                                         guidance=args.teacher_guidance,
                                         image_size=H)
                    m3_teacher_images.append(img)
                t_m3 = time.time() - t1
                m3_teacher_lf = score_lf(m3_teacher_images, verifier)
                m3_teacher_ge = geneval.score_batch(m3_teacher_images, pinfo)
                # [HF-SWAP 2026-04-29] If hf_objective != geneval, score the K
                # M3 survivors with the chosen v_HF and select by it.
                if hf_verifier is not None:
                    m3_teacher_hf = _hf_pool_scores(m3_teacher_images, prompt)
                    best_local = _argmax_hf(m3_teacher_hf, m3_teacher_lf, target_axes)
                else:
                    best_local = _argmax_ge(m3_teacher_ge, m3_teacher_lf,
                                              target_axes, tiebreaker="lf_l1_min")
                img = m3_teacher_images[best_local]
                lf = m3_teacher_lf[best_local]
                gv = m3_teacher_ge[best_local]
                picked_global_idx = int(topk[best_local])
                row = _row(prompt_id, pinfo, t_name, t_vals, "m3", None,
                            picked_global_idx, args.seed, img, lf, gv,
                            target_axes, images_dir,
                            nfe=N_p * args.proxy_steps + K * args.teacher_steps,
                            wall_clock_s=(t_proxy / N_p) * N_p + t_m3,
                            topk_proxy_indices=[int(i) for i in topk])
                _write_row(results_path, row)

            # --- m_tseq Teacher-only sequential LF→HF cascade ---
            # Render N_t teacher noises (already in teacher_pool[:N_t]),
            # screen by LF L1, keep top-K_LF, argmax HF among survivors.
            # NFE = N_t * teacher_steps. No proxy in selection path.
            if "m_tseq" in methods and (prompt_id, t_name, "m_tseq", -1.0) not in done:
                K_LF = args.top_k_lf
                sub_lf = teacher_pool_lf[:N_t]
                l1s = np.array(l1_list(sub_lf))
                topk_lf = np.argsort(l1s)[:K_LF]
                sub_lf_topk = [sub_lf[int(i)] for i in topk_lf]
                if teacher_pool_hf is not None:
                    sub_hf = teacher_pool_hf[:N_t]
                    sub_hf_topk = sub_hf[topk_lf]
                    best_local = _argmax_hf(sub_hf_topk, sub_lf_topk, target_axes)
                else:
                    sub_ge = teacher_pool_geneval[:N_t]
                    sub_ge_topk = [sub_ge[int(i)] for i in topk_lf]
                    best_local = _argmax_ge(sub_ge_topk, sub_lf_topk, target_axes,
                                             tiebreaker="lf_l1_min")
                best_idx = int(topk_lf[best_local])
                img = teacher_pool_images[best_idx]
                lf = teacher_pool_lf[best_idx]
                gv = teacher_pool_geneval[best_idx]
                row = _row(prompt_id, pinfo, t_name, t_vals, "m_tseq", None, best_idx,
                            args.seed, img, lf, gv, target_axes, images_dir,
                            nfe=N_t * args.teacher_steps,
                            wall_clock_s=(t_teacher_shared / Nt_plus_K) * N_t,
                            topk_proxy_indices=[int(i) for i in topk_lf])
                _write_row(results_path, row)

        print(f"  elapsed: {(time.time() - t_all) / 60:.1f} min")

    print(f"\n=== DONE in {(time.time() - t_all) / 60:.1f} min ===")
    print(f"Results: {results_path}")
    print(f"Probe: {probe_path}")


def _argmax_ge(ge_list: List[Dict], lf_list: List[Dict],
                target_axes: Dict[str, float],
                tiebreaker: str = "lf_l1_min") -> int:
    """Argmax by (GenEval success, GenEval confidence, -LF L1)."""
    primary = np.array([1.0 if r["success"] else 0.0 for r in ge_list])
    secondary = np.array([r.get("confidence", 0.0) for r in ge_list])
    l1s = np.array([lf_l1(lf, target_axes) for lf in lf_list])
    # lexsort: last key is primary
    order = np.lexsort((l1s, -secondary, -primary))  # ascending; so -primary flips
    return int(order[0])


def _argmax_hf(hf_scores: np.ndarray, lf_list: List[Dict],
                target_axes: Dict[str, float]) -> int:
    """Argmax by v_HF (continuous), tiebreak by -LF L1.
    Used when --hf-objective is pickscore/hpsv2.
    """
    primary = np.asarray(hf_scores, dtype=np.float64)
    l1s = np.array([lf_l1(lf, target_axes) for lf in lf_list])
    order = np.lexsort((l1s, -primary))  # primary descending; tiebreak l1 ascending
    return int(order[0])


def _row(prompt_id, pinfo, t_name, t_vals, method, lam, picked_idx,
         base_seed, img, lf, gv, target_axes, images_dir, nfe, wall_clock_s,
         topk_proxy_indices=None, postprocess=False):
    # Save image
    lam_str = f"lam{lam:.2f}" if lam is not None else "none"
    fname = f"{prompt_id}__{t_name}__{method}__{lam_str}.png"
    img_path = images_dir / fname
    if not img_path.exists():
        img.save(img_path)
    return {
        "prompt_id": prompt_id,
        "prompt": pinfo["prompt"],
        "prompt_category": pinfo["category"],
        "prompt_objects": pinfo.get("objects", []),
        "lf_target": t_name,
        "lf_target_values": t_vals,
        "lf_target_axes": target_axes,
        "method": method,
        "lambda": lam if lam is not None else -1.0,
        "picked_noise_idx": picked_idx,
        "base_seed": base_seed,
        "lf_scores": lf,
        "lf_l1_to_target": lf_l1(lf, target_axes),
        "geneval_success": bool(gv["success"]),
        "geneval_confidence": float(gv.get("confidence", 0.0)),
        "geneval_detected": gv.get("detected", {}),
        "nfe": nfe,
        "wall_clock_s": wall_clock_s,
        "image_path": str(img_path),
        "postprocess_applied": postprocess,
        "topk_proxy_indices": topk_proxy_indices,
    }


def _write_row(path: Path, row: Dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--num-noises-proxy", type=int, default=64)
    ap.add_argument("--num-noises-teacher", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-k-lf", type=int, default=8,
                    help="m_tseq: top-K survivors after LF screen on N_t teacher renders.")
    ap.add_argument("--lambda-sweep", type=str, default="0,0.1,0.5,1.0,2.0")
    ap.add_argument("--lf-targets", type=str,
                    default="bright_high_contrast,dark_moody")
    ap.add_argument("--methods", type=str,
                    default="m0,m1,m1p5,m2,m3,m3alt")
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--no-stratified", dest="stratified", action="store_false")
    ap.set_defaults(stratified=True)
    ap.add_argument("--geneval-prompts", type=str,
                    default="benchmarks/geneval/prompts/evaluation_metadata.jsonl")
    ap.add_argument("--geneval-threshold", type=float, default=0.1,
                    help="OwlViT detection threshold; unused when --scorer=native")
    ap.add_argument("--scorer", type=str, default="native",
                    choices=["native", "owlvit"],
                    help="native=Mask2Former+CLIP (official GenEval pipeline); owlvit=proxy")
    ap.add_argument("--prompt-pin", type=str, default=None,
                    help="Path to pinned prompt list JSON. "
                         "Overrides stratified sampling.")
    ap.add_argument("--prompt-start", type=int, default=0,
                    help="Slice start index into selected prompts (for sharding).")
    ap.add_argument("--prompt-end", type=int, default=None,
                    help="Slice end index (exclusive). None=all.")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--proxy-steps", type=int, default=4)
    ap.add_argument("--proxy-guidance", type=float, default=0.0)
    ap.add_argument("--teacher-steps", type=int, default=50)
    ap.add_argument("--teacher-guidance", type=float, default=3.5)
    ap.add_argument("--model-family", type=str, default="flux",
                    choices=["flux", "sd3", "sdxl"],
                    help="Teacher--proxy pair family.")
    ap.add_argument("--hf-objective", type=str, default="geneval",
                    choices=["geneval", "pickscore", "hpsv2"],
                    help="HF objective for M1/M2/M3 selection. "
                         "geneval (default): GenEval pass + confidence; "
                         "pickscore / hpsv2: learned preference scalar. "
                         "GenEval is always logged regardless of selection.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str,
                    default="outputs/joint_geneval_lf/micro")
    return ap.parse_args()


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARNING: PYTHONHASHSEED is not 0 — noise_from_id reproducibility "
              "depends on this. Set PYTHONHASHSEED=0 before launching.",
              file=sys.stderr)
    args = parse_args()
    run_pilot(args)
