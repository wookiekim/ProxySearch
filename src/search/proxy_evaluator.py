"""Batch evaluation of candidates using proxy model."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DiffusionPipeline, FluxPipeline
from PIL import Image
from tqdm import tqdm

from ..models.verifier_loader import VerifierEnsemble


class ProxyEvaluator:
    """
    Batch evaluation of candidate noise vectors using a proxy model.

    This class handles the efficient evaluation of many candidate noise
    vectors by batching them through the proxy model and scoring with
    the verifier ensemble.
    """

    def __init__(
        self,
        proxy_pipeline: DiffusionPipeline,
        verifier: VerifierEnsemble,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        batch_size: int = 8,
        image_size: int = 1024,
    ):
        """
        Initialize the proxy evaluator.

        Args:
            proxy_pipeline: Loaded diffusers pipeline for proxy model
                           (e.g., FLUX-schnell or SD3.5-Large-Turbo).
            verifier: VerifierEnsemble instance for scoring generated images.
            num_inference_steps: Number of denoising steps (default 4 for proxy).
            guidance_scale: Classifier-free guidance scale (0.0 for schnell).
            batch_size: Number of candidates to process in parallel.
                       Limited by GPU memory.
            image_size: Target image size (assumes square images).
        """
        self.proxy_pipeline = proxy_pipeline
        self.verifier = verifier
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.image_size = image_size

    def evaluate(
        self,
        noises: torch.Tensor,
        prompt: str,
        return_timing: bool = False,
    ) -> Tuple[List[Image.Image], np.ndarray, Optional[Dict[str, float]]]:
        """
        Evaluate all candidate noises using the proxy model.

        Args:
            noises: N candidate initial noise vectors of shape (N, C, H, W).
            prompt: Text prompt for generation.
            return_timing: Whether to return timing breakdown.

        Returns:
            Tuple of:
                - images: List of N generated PIL images.
                - scores: Array of N verifier scores.
                - timing: Dict of timing info (only if return_timing=True).
        """
        N = noises.shape[0]
        all_images = []
        timing = {"generation": 0.0, "scoring": 0.0} if return_timing else None

        # Process in batches
        for batch_start in range(0, N, self.batch_size):
            batch_end = min(batch_start + self.batch_size, N)
            batch_noises = noises[batch_start:batch_end]

            # Generate images
            t_gen_start = time.time()
            batch_images = self._generate_batch(batch_noises, prompt)
            if return_timing:
                timing["generation"] += time.time() - t_gen_start

            all_images.extend(batch_images)

            # Clear cache between batches to manage memory
            torch.cuda.empty_cache()

        # Score all images at once (more efficient for verifiers)
        t_score_start = time.time()
        scores = self.verifier(all_images, prompt)
        if return_timing:
            timing["scoring"] = time.time() - t_score_start

        if return_timing:
            return all_images, scores, timing
        return all_images, scores, None

    def _pack_latents_for_flux(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack (B, C, H, W) latents into FLUX's (B, H*W/4, C*4) format."""
        pipe = self.proxy_pipeline
        B, C, H, W = latents.shape
        return pipe._pack_latents(latents, B, C, H, W)

    def _prepare_latents(self, noises: torch.Tensor) -> torch.Tensor:
        """Prepare latents for the pipeline, handling FLUX packing if needed."""
        if isinstance(self.proxy_pipeline, FluxPipeline):
            return self._pack_latents_for_flux(noises)
        return noises

    def _generate_batch(
        self,
        noises: torch.Tensor,
        prompt: str,
    ) -> List[Image.Image]:
        """
        Generate images for a batch of noise vectors.

        Args:
            noises: Batch of noise vectors of shape (B, C, H, W).
            prompt: Text prompt for generation.

        Returns:
            List of B generated PIL images.
        """
        batch_size = noises.shape[0]

        # Create batch of prompts
        prompts = [prompt] * batch_size

        # Pack latents for FLUX if needed
        latents = self._prepare_latents(noises)

        # Run inference
        with torch.no_grad():
            output = self.proxy_pipeline(
                prompt=prompts,
                latents=latents,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                output_type="pil",
            )

        return output.images

    def evaluate_with_progress(
        self,
        noises: torch.Tensor,
        prompt: str,
        desc: str = "Evaluating candidates",
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Evaluate candidates with progress bar.

        Args:
            noises: N candidate initial noise vectors of shape (N, C, H, W).
            prompt: Text prompt for generation.
            desc: Description for progress bar.

        Returns:
            Tuple of (images, scores).
        """
        N = noises.shape[0]
        all_images = []

        pbar = tqdm(range(0, N, self.batch_size), desc=desc)
        for batch_start in pbar:
            batch_end = min(batch_start + self.batch_size, N)
            batch_noises = noises[batch_start:batch_end]

            batch_images = self._generate_batch(batch_noises, prompt)
            all_images.extend(batch_images)

            pbar.set_postfix({"processed": batch_end})
            torch.cuda.empty_cache()

        scores = self.verifier(all_images, prompt)
        return all_images, scores

    def evaluate_single(
        self,
        noise: torch.Tensor,
        prompt: str,
    ) -> Tuple[Image.Image, float]:
        """
        Evaluate a single noise vector.

        Args:
            noise: Single noise vector of shape (1, C, H, W).
            prompt: Text prompt for generation.

        Returns:
            Tuple of (generated image, verifier score).
        """
        if noise.dim() == 3:
            noise = noise.unsqueeze(0)

        images, scores, _ = self.evaluate(noise, prompt)
        return images[0], float(scores[0])

    def get_individual_scores(
        self,
        images: List[Image.Image],
        prompt: str,
    ) -> Dict[str, np.ndarray]:
        """
        Get per-verifier scores for a set of images.

        Args:
            images: List of PIL images.
            prompt: Text prompt used for generation.

        Returns:
            Dictionary mapping verifier names to score arrays.
        """
        return self.verifier.score_individual(images, prompt)

    def compute_statistics(
        self,
        scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute statistics over candidate scores.

        Args:
            scores: Array of candidate scores.

        Returns:
            Dictionary with statistics (mean, std, max, min, median).
        """
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
            "median": float(np.median(scores)),
        }
