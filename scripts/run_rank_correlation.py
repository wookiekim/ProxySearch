#!/usr/bin/env python3
"""
Script for running the proxy-teacher rank correlation study.

Usage:
    python scripts/run_rank_correlation.py --num-prompts 100 --num-noises 64
    python scripts/run_rank_correlation.py --config configs/flux_schnell_dev.yaml
    python scripts/run_rank_correlation.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run proxy-teacher rank correlation study"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/flux_schnell_dev.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to test",
    )
    parser.add_argument(
        "--num-noises",
        type=int,
        default=64,
        help="Number of noise vectors per prompt",
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
        default="outputs/correlation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verifiers",
        type=str,
        default=None,
        help="Comma-separated verifier names (e.g. 'brightness,compressibility'). Overrides config.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size from config (e.g. 512 for faster runs)",
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
    print("DRY RUN: Testing rank correlation components")
    print("=" * 60)

    # Test correlation computation
    print("\n[1/3] Testing Spearman correlation...")
    proxy_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    teacher_scores = np.array([0.15, 0.35, 0.45, 0.75, 0.85])

    rho, p_value = stats.spearmanr(proxy_scores, teacher_scores)
    print(f"  Proxy scores: {proxy_scores}")
    print(f"  Teacher scores: {teacher_scores}")
    print(f"  Spearman rho: {rho:.4f}")
    print(f"  p-value: {p_value:.4f}")
    assert rho > 0.8, f"Expected high correlation, got {rho}"
    print("  PASS")

    # Test Pearson correlation
    print("\n[2/3] Testing Pearson correlation...")
    r, p_value = stats.pearsonr(proxy_scores, teacher_scores)
    print(f"  Pearson r: {r:.4f}")
    assert r > 0.8, f"Expected high correlation, got {r}"
    print("  PASS")

    # Test RankCorrelationStudy class import
    print("\n[3/3] Testing RankCorrelationStudy import...")
    from src.analysis.rank_correlation import RankCorrelationStudy
    print("  RankCorrelationStudy class available")
    print("  PASS")

    print("\n" + "=" * 60)
    print("DRY RUN OK")
    print("=" * 60)


def generate_test_prompts(num_prompts: int) -> list:
    """Generate test prompts for correlation study."""
    # Diverse set of compositional prompts
    templates = [
        "a {} {} on a {} {}",
        "a {} and a {} in a {}",
        "{} {} sitting next to a {}",
        "a {} colored {} with {} {}",
        "{} people {} in a {}",
    ]

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink"]
    objects = ["apple", "ball", "cup", "book", "flower", "bird", "cat", "dog"]
    materials = ["wooden", "metal", "glass", "stone", "plastic"]
    locations = ["table", "floor", "shelf", "garden", "room", "park"]
    actions = ["standing", "playing", "running", "sitting", "walking"]

    np.random.seed(42)
    prompts = []

    for i in range(num_prompts):
        template = np.random.choice(templates)
        words = []

        # Fill template based on its structure
        for _ in range(template.count("{}")):
            category = np.random.choice(["color", "object", "material", "location", "action", "number"])
            if category == "color":
                words.append(np.random.choice(colors))
            elif category == "object":
                words.append(np.random.choice(objects))
            elif category == "material":
                words.append(np.random.choice(materials))
            elif category == "location":
                words.append(np.random.choice(locations))
            elif category == "action":
                words.append(np.random.choice(actions))
            elif category == "number":
                words.append(str(np.random.randint(2, 6)))

        prompt = template.format(*words)
        prompts.append({
            "prompt": prompt,
            "prompt_id": f"corr_{i:04d}",
        })

    return prompts


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.dry_run:
        run_dry_run()
        return

    # Load config
    config = load_config(args.config)

    # Set up logging
    logger = setup_logging(
        log_dir=config.get("log_dir", "outputs/logs"),
        experiment_name="rank_correlation",
    )

    set_seed(args.seed)

    # Load or generate prompts
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = json.load(f)
    else:
        prompts = generate_test_prompts(args.num_prompts)

    prompts = prompts[:args.num_prompts]
    logger.info(f"Using {len(prompts)} prompts with {args.num_noises} noises each")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    from src.models.proxy_loader import ProxyLoader
    from src.models.teacher_loader import TeacherLoader
    from src.models.verifier_loader import VerifierEnsemble
    from src.search.noise_sampler import NoiseSampler
    from src.analysis.rank_correlation import RankCorrelationStudy

    # Get model paths safely
    model_paths = config.get("model_paths", {}) or {}
    model_family = config.get("model_family", "flux")

    # Load verifier (CLI override takes precedence)
    if args.verifiers:
        verifier_list = [v.strip() for v in args.verifiers.split(",")]
    else:
        verifier_list = config.get("verifiers", ["clip", "image_reward"])

    verifier = VerifierEnsemble(
        verifiers=verifier_list,
        weights=config.get("verifier_weights"),
    )

    # Initialize loaders (models loaded/unloaded sequentially to save GPU memory)
    proxy_loader = ProxyLoader(
        model_family=model_family,
        model_path=model_paths.get("proxy") if model_paths else None,
        lcm_lora_path=model_paths.get("lcm_lora") if model_paths else None,
        proxy_unet_override_path=model_paths.get("proxy_unet_override") if model_paths else None,
    )
    teacher_loader = TeacherLoader(
        model_family=model_family,
        model_path=model_paths.get("teacher") if model_paths else None,
        teacher_lora_path=model_paths.get("teacher_lora_path") if model_paths else None,
    )

    # Initialize noise sampler
    image_size = args.image_size or config.get("image_size", 1024)
    latent_shape = proxy_loader.get_latent_shape(image_size)
    noise_sampler = NoiseSampler(latent_shape=latent_shape)

    # Run correlation study with sequential model loading
    study = RankCorrelationStudy(
        proxy_pipeline=None,  # loaded sequentially inside run_study_sequential
        teacher_pipeline=None,
        verifier=verifier,
        noise_sampler=noise_sampler,
    )

    prompt_strings = [p["prompt"] for p in prompts]

    logger.info(f"Running sequential correlation study: {len(prompt_strings)} prompts, {args.num_noises} noises, {image_size}px")

    results = study.run_study_sequential(
        prompts=prompt_strings,
        proxy_loader=proxy_loader,
        teacher_loader=teacher_loader,
        num_noises=args.num_noises,
        seed=args.seed,
        verbose=True,
        checkpoint_dir=output_dir,
        proxy_steps=config.get("proxy_steps", 4),
        teacher_steps=config.get("teacher_steps", 28),
        proxy_guidance=config.get("proxy_guidance_scale", 0.0),
        teacher_guidance=config.get("guidance_scale", 3.5),
        teacher_true_cfg_scale=config.get("true_cfg_scale", 1.0),
        image_size=image_size,
    )

    # Save results
    study.save_results(output_dir / "correlation_results.json")

    # Generate plots
    try:
        study.plot_correlation_distribution(
            save_path=output_dir / "correlation_distribution.png"
        )
        if len(study.results) > 0:
            study.plot_correlation_scatter(
                prompt_idx=0,
                save_path=output_dir / "correlation_scatter_example.png",
            )
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # Print summary
    print("\n" + "=" * 60)
    print("RANK CORRELATION STUDY RESULTS")
    print("=" * 60)
    print(study.generate_table())
    print("=" * 60)

    logger.info(f"Results saved to: {output_dir}")

    # Cleanup
    verifier.unload()


if __name__ == "__main__":
    main()
