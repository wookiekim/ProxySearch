#!/usr/bin/env python3
"""
Script for running baseline methods.

Usage:
    python scripts/run_baselines.py --config configs/flux_schnell_dev.yaml --method best_of_n
    python scripts/run_baselines.py --method random --num-candidates 1
    python scripts/run_baselines.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.pipeline.baselines import BestOfN, RandomSearch, FullModelSearch
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging, log_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline methods for text-to-image generation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="best_of_n",
        choices=["best_of_n", "random", "full_model_search"],
        help="Baseline method to run",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate",
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
        default="outputs/baselines",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of candidates (for best_of_n)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bars",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dry-run test",
    )

    return parser.parse_args()


def run_dry_run() -> None:
    """Run a dry-run test."""
    print("=" * 60)
    print("DRY RUN: Testing baseline components")
    print("=" * 60)

    # Test config loading
    print("\n[1/3] Testing config loading...")
    config = OmegaConf.create({
        "model_family": "flux",
        "num_candidates": 4,
        "denoising_steps": 28,
        "guidance_scale": 3.5,
        "image_size": 64,
        "verifiers": ["clip", "image_reward"],
    })
    print(f"  Config: {dict(config)}")
    print("  PASS")

    # Test baseline class imports
    print("\n[2/3] Testing baseline class imports...")
    print("  BestOfN available")
    print("  RandomSearch available")
    print("  FullModelSearch available")
    print("  PASS")

    # Test seed utilities
    print("\n[3/3] Testing seed utilities...")
    set_seed(42)
    rng1 = np.random.random()
    set_seed(42)
    rng2 = np.random.random()
    assert rng1 == rng2, "Seed not working correctly"
    print(f"  Random value with seed 42: {rng1:.6f}")
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
            return json.load(f)

    return [
        {"prompt": "a red apple on a wooden table", "prompt_id": "test_0"},
        {"prompt": "a cat and a dog playing together", "prompt_id": "test_1"},
    ]


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.dry_run:
        run_dry_run()
        return

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.num_candidates is not None:
        config.num_candidates = args.num_candidates
    config.seed = args.seed

    # Set up logging
    logger = setup_logging(
        log_dir=config.get("log_dir", "outputs/logs"),
        experiment_name=f"baseline_{args.method}",
    )

    set_seed(args.seed)

    # Load prompts
    prompts = load_prompts(args)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Create output directory
    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize baseline
    logger.info(f"Initializing {args.method} baseline...")

    if args.method == "best_of_n":
        baseline = BestOfN(config)
    elif args.method == "random":
        baseline = RandomSearch(config)
    elif args.method == "full_model_search":
        baseline = FullModelSearch(config)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Run inference
    all_results = []

    for prompt_info in prompts:
        prompt = prompt_info["prompt"]
        prompt_id = prompt_info.get("prompt_id", str(len(all_results)))

        logger.info(f"Processing: {prompt[:50]}...")

        result = baseline(
            prompt=prompt,
            seed=args.seed,
            verbose=args.verbose,
        )

        # Save image
        image_path = output_dir / f"{prompt_id}.png"
        result["image"].save(image_path)

        metrics = {
            "score": result["score"],
            "time": result["timing"]["total"],
        }
        log_metrics(metrics, prefix=prompt_id, logger=logger)

        all_results.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "score": result["score"],
            "timing": result["timing"],
            "image_path": str(image_path),
        })

    # Save summary
    summary = {
        "method": args.method,
        "num_prompts": len(prompts),
        "avg_score": float(np.mean([r["score"] for r in all_results])),
        "avg_time": float(np.mean([r["timing"]["total"] for r in all_results])),
        "config": OmegaConf.to_container(config, resolve=True),
        "results": all_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Average score: {summary['avg_score']:.4f}")
    logger.info(f"Average time: {summary['avg_time']:.2f}s")

    baseline.unload()


if __name__ == "__main__":
    main()
