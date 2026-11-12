"""PartiPrompts benchmark evaluator."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .metrics import clip_score, image_reward_score


class PartiEvaluator:
    """
    Evaluator for the PartiPrompts benchmark.

    PartiPrompts contains 1632 prompts across multiple challenge categories
    designed to comprehensively test prompt-following capabilities:
    - Abstract
    - Arts
    - Food & Beverage
    - Illustrations
    - Indoor Scenes
    - Outdoor Scenes
    - People
    - Produce & Plants
    - Vehicles
    - World Knowledge
    - And more...

    Evaluation uses CLIP score, ImageReward, and optionally FID.
    """

    CHALLENGE_CATEGORIES = [
        "Simple Detail",
        "Fine-grained Detail",
        "Properties & Positioning",
        "Basic",
        "Complex",
        "Imagination",
        "Quantity",
        "Style & Format",
        "World Knowledge",
        "Writing & Symbols",
    ]

    def __init__(
        self,
        prompts_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ):
        """
        Initialize the PartiPrompts evaluator.

        Args:
            prompts_path: Path to PartiPrompts file.
            device: Target device for metric computation.
        """
        self.prompts_path = prompts_path
        self.device = device
        self._prompts: Optional[List[Dict]] = None

        # Lazy loading for metrics
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._ir_model = None

    def load_prompts(self, path: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Load PartiPrompts from file.

        Args:
            path: Path to prompts file.

        Returns:
            List of prompt dictionaries.
        """
        path = path or self.prompts_path
        if path is None:
            raise ValueError("No prompts path specified")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Handle different file formats
        if isinstance(data, list):
            self._prompts = data
        elif isinstance(data, dict) and "prompts" in data:
            self._prompts = data["prompts"]
        else:
            raise ValueError("Unknown PartiPrompts file format")

        return self._prompts

    def get_prompts(
        self,
        category: Optional[str] = None,
        challenge: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get prompts with optional filtering.

        Args:
            category: Content category filter.
            challenge: Challenge category filter.

        Returns:
            List of prompt dictionaries.
        """
        if self._prompts is None:
            self.load_prompts()

        prompts = self._prompts

        if category is not None:
            prompts = [p for p in prompts if p.get("category") == category]

        if challenge is not None:
            prompts = [p for p in prompts if p.get("challenge") == challenge]

        return prompts

    def _ensure_clip_loaded(self) -> None:
        """Ensure CLIP model is loaded."""
        if self._clip_model is not None:
            return

        try:
            import open_clip
        except ImportError:
            raise ImportError("open_clip required. Install with: pip install open-clip-torch")

        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            device=self.device,
        )
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self._clip_model.eval()

    def _ensure_ir_loaded(self) -> None:
        """Ensure ImageReward model is loaded."""
        if self._ir_model is not None:
            return

        try:
            import ImageReward as ir
        except ImportError:
            raise ImportError("ImageReward required. Install with: pip install image-reward")

        self._ir_model = ir.load("ImageReward-v1.0", device=self.device)

    def evaluate_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Dict[str, float]:
        """
        Evaluate a single image.

        Args:
            image: Generated image.
            prompt: Text prompt.

        Returns:
            Dictionary with CLIP and ImageReward scores.
        """
        self._ensure_clip_loaded()
        self._ensure_ir_loaded()

        clip = clip_score(
            image,
            prompt,
            model=self._clip_model,
            preprocess=self._clip_preprocess,
            tokenizer=self._clip_tokenizer,
            device=self.device,
        )

        ir = image_reward_score(
            image,
            prompt,
            model=self._ir_model,
            device=self.device,
        )

        return {
            "clip_score": float(clip),
            "image_reward": float(ir),
        }

    def evaluate_batch(
        self,
        images: List[Image.Image],
        prompt_infos: List[Dict],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of images.

        Args:
            images: List of generated images.
            prompt_infos: List of prompt dictionaries.

        Returns:
            Dictionary with per-image and aggregate metrics.
        """
        self._ensure_clip_loaded()
        self._ensure_ir_loaded()

        prompts = [p.get("prompt", "") for p in prompt_infos]

        # Compute scores
        clip_scores = clip_score(
            images,
            prompts,
            model=self._clip_model,
            preprocess=self._clip_preprocess,
            tokenizer=self._clip_tokenizer,
            device=self.device,
        )

        ir_scores = image_reward_score(
            images,
            prompts,
            model=self._ir_model,
            device=self.device,
        )

        # Per-image results
        results = []
        for i, prompt_info in enumerate(prompt_infos):
            results.append({
                "prompt_id": prompt_info.get("prompt_id", str(i)),
                "category": prompt_info.get("category"),
                "challenge": prompt_info.get("challenge"),
                "clip_score": float(clip_scores[i]),
                "image_reward": float(ir_scores[i]),
            })

        # Per-challenge aggregation
        challenge_results = {}
        for challenge in self.CHALLENGE_CATEGORIES:
            ch_results = [r for r in results if r.get("challenge") == challenge]
            if ch_results:
                challenge_results[challenge] = {
                    "clip_mean": float(np.mean([r["clip_score"] for r in ch_results])),
                    "clip_std": float(np.std([r["clip_score"] for r in ch_results])),
                    "ir_mean": float(np.mean([r["image_reward"] for r in ch_results])),
                    "ir_std": float(np.std([r["image_reward"] for r in ch_results])),
                    "count": len(ch_results),
                }

        # Overall aggregation
        return {
            "overall": {
                "clip_mean": float(np.mean(clip_scores)),
                "clip_std": float(np.std(clip_scores)),
                "ir_mean": float(np.mean(ir_scores)),
                "ir_std": float(np.std(ir_scores)),
                "count": len(images),
            },
            "per_challenge": challenge_results,
            "detailed_results": results,
        }

    def compute_summary_statistics(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute summary statistics from evaluation results.

        Args:
            results: Results from evaluate_batch.

        Returns:
            Dictionary with summary statistics.
        """
        detailed = results.get("detailed_results", [])

        if not detailed:
            return {}

        clip_scores = [r["clip_score"] for r in detailed]
        ir_scores = [r["image_reward"] for r in detailed]

        return {
            "clip_mean": float(np.mean(clip_scores)),
            "clip_std": float(np.std(clip_scores)),
            "clip_median": float(np.median(clip_scores)),
            "clip_p25": float(np.percentile(clip_scores, 25)),
            "clip_p75": float(np.percentile(clip_scores, 75)),
            "ir_mean": float(np.mean(ir_scores)),
            "ir_std": float(np.std(ir_scores)),
            "ir_median": float(np.median(ir_scores)),
            "ir_p25": float(np.percentile(ir_scores, 25)),
            "ir_p75": float(np.percentile(ir_scores, 75)),
        }

    def unload(self) -> None:
        """Unload models from memory."""
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._ir_model = None
        torch.cuda.empty_cache()
