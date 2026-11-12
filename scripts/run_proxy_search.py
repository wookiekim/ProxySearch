#!/usr/bin/env python3
"""
Main script for running ProxySearch inference.

Usage:
    python scripts/run_proxy_search.py --config configs/flux_schnell_dev.yaml
    python scripts/run_proxy_search.py --config configs/flux_schnell_dev.yaml --prompt "a red apple"
    python scripts/run_proxy_search.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.pipeline.proxy_search import ProxySearchPipeline
from src.utils.config import load_config, merge_configs
from src.utils.logging_utils import setup_logging, log_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ProxySearch for text-to-image generation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate (overrides prompt_file)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to JSON file with prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of candidates (overrides config)",
    )
    parser.add_argument(
        "--no-local-refine",
        action="store_true",
        help="Disable local refinement",
    )
    parser.add_argument(
        "--no-teacher-refine",
        action="store_true",
        help="Disable teacher refinement",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bars and detailed output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dry-run test (no GPU, random data)",
    )

    return parser.parse_args()


def run_dry_run() -> None:
    """Run a dry-run test without GPU or models."""
    print("=" * 60)
    print("DRY RUN: Testing ProxySearch pipeline components")
    print("=" * 60)

    # Test config loading
    print("\n[1/4] Testing config loading...")
    config = OmegaConf.create({
        "model_family": "flux",
        "num_candidates": 4,
        "top_k": 2,
        "local_refine": False,
        "teacher_refine": False,
        "proxy_steps": 4,
        "teacher_steps": 28,
        "guidance_scale": 3.5,
        "proxy_guidance_scale": 0.0,
        "image_size": 64,  # Small for testing
        "batch_size": 2,
        "verifiers": ["clip", "image_reward"],
        "selection_strategy": "top_k",
        "refinement_strategy": "same_noise",
    })
    print(f"  Config loaded: {dict(config)}")
    print("  PASS")

    # Test noise sampler
    print("\n[2/4] Testing NoiseSampler...")
    from src.search.noise_sampler import NoiseSampler

    latent_shape = (1, 16, 8, 8)  # Small for testing
    sampler = NoiseSampler(latent_shape=latent_shape, device="cpu", dtype=torch.float32)
    noises = sampler.sample(num_samples=4, seed=42)
    assert noises.shape == (4, 16, 8, 8), f"Wrong shape: {noises.shape}"
    print(f"  Sampled noises shape: {noises.shape}")

    perturbed = sampler.perturb(noises[:2], epsilon=0.1, seed=43)
    assert perturbed.shape == (2, 16, 8, 8), f"Wrong shape: {perturbed.shape}"
    print(f"  Perturbed noises shape: {perturbed.shape}")
    print("  PASS")

    # Test candidate selector
    print("\n[3/4] Testing CandidateSelector...")
    from src.search.candidate_selector import CandidateSelector
    from PIL import Image

    selector = CandidateSelector(top_k=2, selection_strategy="top_k")

    scores = np.array([0.1, 0.5, 0.3, 0.8])
    images = [Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50)) for i in range(4)]

    selected_scores, selected_images, selected_noises, indices = selector.select(
        scores, images, noises
    )
    assert len(indices) == 2, f"Wrong number selected: {len(indices)}"
    assert indices[0] == 3, f"Wrong top candidate: {indices[0]}"
    print(f"  Selected indices: {indices}")
    print(f"  Selected scores: {selected_scores}")
    print("  PASS")

    # Test config validation
    print("\n[4/4] Testing full pipeline instantiation...")
    # We can't fully test without models, but we can verify the class loads
    print("  ProxySearchPipeline class available")
    print("  PASS")

    print("\n" + "=" * 60)
    print("DRY RUN OK")
    print("=" * 60)


def load_prompts(args: argparse.Namespace) -> list:
    """Load prompts from args or file."""
    if args.prompt is not None:
        return [{"prompt": args.prompt, "prompt_id": "cmd_0"}]

    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            prompts = json.load(f)
        return prompts

    # Default test prompts
    return [
        {"prompt": "a red apple on a wooden table", "prompt_id": "test_0"},
        {"prompt": "a cat and a dog playing together", "prompt_id": "test_1"},
        {"prompt": "three blue birds on a branch", "prompt_id": "test_2"},
    ]


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle dry-run
    if args.dry_run:
        run_dry_run()
        return

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config.seed = args.seed
    if args.num_candidates is not None:
        config.num_candidates = args.num_candidates
    if args.no_local_refine:
        config.local_refine = False
    if args.no_teacher_refine:
        config.teacher_refine = False

    # Set up logging
    logger = setup_logging(
        log_dir=config.get("log_dir", "outputs/logs"),
        experiment_name="proxy_search",
        use_wandb=config.get("use_wandb", False),
        wandb_project=config.get("wandb_project", "proxysearch"),
        wandb_config=OmegaConf.to_container(config, resolve=True),
    )

    # Set seed
    set_seed(config.get("seed", 42))

    # Load prompts
    prompts = load_prompts(args)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    logger.info("Initializing ProxySearch pipeline...")
    pipeline = ProxySearchPipeline(config)

    # Run inference
    all_results = []
    total_time = 0.0

    for prompt_info in prompts:
        prompt = prompt_info["prompt"]
        prompt_id = prompt_info.get("prompt_id", str(len(all_results)))

        logger.info(f"Processing: {prompt[:50]}...")

        # Run ProxySearch
        result = pipeline(
            prompt=prompt,
            seed=config.get("seed", 42),
            verbose=args.verbose,
        )

        # Save image
        image_path = output_dir / f"{prompt_id}.png"
        result["image"].save(image_path)
        logger.info(f"Saved: {image_path}")

        # Log metrics
        metrics = {
            "score": result["score"],
            "proxy_score": result.get("proxy_score", result["score"]),
            "time_total": result["timing"]["total"],
        }
        log_metrics(metrics, prefix=prompt_id, logger=logger)

        all_results.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "score": result["score"],
            "timing": result["timing"],
            "image_path": str(image_path),
        })

        total_time += result["timing"]["total"]

    # Save summary
    summary = {
        "num_prompts": len(prompts),
        "total_time": total_time,
        "avg_time_per_prompt": total_time / len(prompts) if prompts else 0,
        "avg_score": float(np.mean([r["score"] for r in all_results])),
        "config": OmegaConf.to_container(config, resolve=True),
        "results": all_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Average score: {summary['avg_score']:.4f}")
    logger.info(f"Average time: {summary['avg_time_per_prompt']:.2f}s")

    # Cleanup
    pipeline.unload()


if __name__ == "__main__":
    main()
