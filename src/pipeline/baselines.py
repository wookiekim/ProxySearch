"""Baseline methods for comparison with ProxySearch."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline, FluxPipeline
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from ..models.teacher_loader import TeacherLoader
from ..models.verifier_loader import VerifierEnsemble
from ..search.noise_sampler import NoiseSampler


class BestOfN:
    """
    Best-of-N baseline using full teacher model.

    This is the standard baseline from Ma et al. (2025): sample N noise
    vectors and evaluate each with the full teacher model, selecting
    the one with highest verifier score.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Initialize the Best-of-N baseline.

        Args:
            config: Configuration including:
                - model_family: "flux" or "sd3"
                - num_candidates: N
                - denoising_steps: T (28 for FLUX, 50 for SDXL)
                - guidance_scale: CFG scale
                - verifiers: List of verifier names
                - verifier_weights: Weights for aggregation
                - image_size: Target image size
        """
        self.config = config

        self.model_family = config.get("model_family", "flux")
        self.num_candidates = config.get("num_candidates", 16)
        self.denoising_steps = config.get("denoising_steps", 28)
        self.guidance_scale = config.get("guidance_scale", 3.5)
        self.image_size = config.get("image_size", 1024)

        self.verifiers = config.get("verifiers", ["clip", "image_reward"])
        self.verifier_weights = config.get("verifier_weights", None)

        self.model_paths = config.get("model_paths", {})

        # Lazy loading
        self._teacher_loader: Optional[TeacherLoader] = None
        self._pipeline: Optional[DiffusionPipeline] = None
        self._verifier: Optional[VerifierEnsemble] = None
        self._noise_sampler: Optional[NoiseSampler] = None

    def _get_latent_shape(self) -> tuple:
        """Return latent shape for model family."""
        latent_size = self.image_size // 8
        if self.model_family == "flux":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sd3":
            return (1, 16, latent_size, latent_size)

    def _prepare_latents(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare latents for the pipeline, handling FLUX packing if needed."""
        if isinstance(self._pipeline, FluxPipeline):
            B, C, H, W = noise.shape
            return self._pipeline._pack_latents(noise, B, C, H, W)
        return noise

    def _ensure_loaded(self) -> None:
        """Ensure all components are loaded."""
        if self._pipeline is None:
            self._teacher_loader = TeacherLoader(
                model_family=self.model_family,
                model_path=self.model_paths.get("teacher") if self.model_paths else None,
            )
            self._pipeline = self._teacher_loader.load()

            self._verifier = VerifierEnsemble(
                verifiers=self.verifiers,
                weights=self.verifier_weights,
            )

            self._noise_sampler = NoiseSampler(
                latent_shape=self._get_latent_shape(),
            )

    def __call__(
        self,
        prompt: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Best-of-N search.

        Args:
            prompt: Text prompt for generation.
            seed: Random seed.
            verbose: Whether to show progress.

        Returns:
            Dictionary with best image, score, timing, etc.
        """
        self._ensure_loaded()

        timing = {}
        t_total_start = time.time()

        # Sample noise vectors
        noises, _ = self._noise_sampler.sample_with_ids(
            num_samples=self.num_candidates,
            base_seed=seed or 0,
            prompt_id=prompt,
        )

        # Evaluate each candidate with full model
        candidates = []
        scores = []

        iterator = range(self.num_candidates)
        if verbose:
            iterator = tqdm(iterator, desc="Best-of-N evaluation")

        t_gen_start = time.time()
        for i in iterator:
            latents = self._prepare_latents(noises[i : i + 1])
            with torch.no_grad():
                output = self._pipeline(
                    prompt=prompt,
                    latents=latents,
                    num_inference_steps=self.denoising_steps,
                    guidance_scale=self.guidance_scale,
                    height=self.image_size,
                    width=self.image_size,
                    output_type="pil",
                )
            image = output.images[0]
            score = float(self._verifier(image, prompt))

            candidates.append(image)
            scores.append(score)

            torch.cuda.empty_cache()

        timing["generation"] = time.time() - t_gen_start

        # Select best
        scores = np.array(scores)
        best_idx = int(np.argmax(scores))

        timing["total"] = time.time() - t_total_start

        return {
            "image": candidates[best_idx],
            "score": float(scores[best_idx]),
            "noise": noises[best_idx : best_idx + 1],
            "all_scores": scores,
            "timing": timing,
            "config": {
                "method": "best_of_n",
                "num_candidates": self.num_candidates,
                "denoising_steps": self.denoising_steps,
            },
        }

    def unload(self) -> None:
        """Unload models from memory."""
        if self._teacher_loader is not None:
            self._teacher_loader.unload()
        if self._verifier is not None:
            self._verifier.unload()
        torch.cuda.empty_cache()


class RandomSearch:
    """
    Random search baseline (single sample, no search).

    Simply generates a single image with a random noise vector.
    This represents the zero-search baseline.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Initialize the random search baseline.

        Args:
            config: Configuration including model and generation settings.
        """
        self.config = config

        self.model_family = config.get("model_family", "flux")
        self.denoising_steps = config.get("denoising_steps", 28)
        self.guidance_scale = config.get("guidance_scale", 3.5)
        self.image_size = config.get("image_size", 1024)

        self.verifiers = config.get("verifiers", ["clip", "image_reward"])
        self.verifier_weights = config.get("verifier_weights", None)

        self.model_paths = config.get("model_paths", {})

        self._teacher_loader: Optional[TeacherLoader] = None
        self._pipeline: Optional[DiffusionPipeline] = None
        self._verifier: Optional[VerifierEnsemble] = None
        self._noise_sampler: Optional[NoiseSampler] = None

    def _get_latent_shape(self) -> tuple:
        """Return latent shape for model family."""
        latent_size = self.image_size // 8
        if self.model_family == "flux":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sd3":
            return (1, 16, latent_size, latent_size)

    def _prepare_latents(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare latents for the pipeline, handling FLUX packing if needed."""
        if isinstance(self._pipeline, FluxPipeline):
            B, C, H, W = noise.shape
            return self._pipeline._pack_latents(noise, B, C, H, W)
        return noise

    def _ensure_loaded(self) -> None:
        """Ensure all components are loaded."""
        if self._pipeline is None:
            self._teacher_loader = TeacherLoader(
                model_family=self.model_family,
                model_path=self.model_paths.get("teacher") if self.model_paths else None,
            )
            self._pipeline = self._teacher_loader.load()

            self._verifier = VerifierEnsemble(
                verifiers=self.verifiers,
                weights=self.verifier_weights,
            )

            self._noise_sampler = NoiseSampler(
                latent_shape=self._get_latent_shape(),
            )

    def __call__(
        self,
        prompt: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a single random image.

        Args:
            prompt: Text prompt for generation.
            seed: Random seed.
            verbose: Whether to show progress (ignored).

        Returns:
            Dictionary with image, score, timing.
        """
        self._ensure_loaded()

        timing = {}
        t_start = time.time()

        # Sample single noise
        noise = self._noise_sampler.sample_single(seed=seed)

        # Generate image
        latents = self._prepare_latents(noise)
        with torch.no_grad():
            output = self._pipeline(
                prompt=prompt,
                latents=latents,
                num_inference_steps=self.denoising_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                output_type="pil",
            )
        image = output.images[0]

        timing["generation"] = time.time() - t_start

        # Score
        score = float(self._verifier(image, prompt))

        timing["total"] = time.time() - t_start

        return {
            "image": image,
            "score": score,
            "noise": noise,
            "timing": timing,
            "config": {
                "method": "random_search",
                "denoising_steps": self.denoising_steps,
            },
        }

    def unload(self) -> None:
        """Unload models from memory."""
        if self._teacher_loader is not None:
            self._teacher_loader.unload()
        if self._verifier is not None:
            self._verifier.unload()
        torch.cuda.empty_cache()


class FullModelSearch:
    """
    Full model search baseline (Ma et al. 2025).

    This is the original noise trajectory search method that uses
    the full teacher model for both search and generation. Supports
    optional local refinement.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Initialize the full model search baseline.

        Args:
            config: Configuration including:
                - model_family: "flux" or "sd3"
                - num_candidates: N
                - denoising_steps: T
                - guidance_scale: CFG scale
                - local_refine: bool
                - local_perturbations: M
                - local_rounds: number of refinement rounds
                - local_epsilon: perturbation strength
        """
        self.config = config

        self.model_family = config.get("model_family", "flux")
        self.num_candidates = config.get("num_candidates", 16)
        self.denoising_steps = config.get("denoising_steps", 28)
        self.guidance_scale = config.get("guidance_scale", 3.5)
        self.image_size = config.get("image_size", 1024)

        self.local_refine = config.get("local_refine", False)
        self.local_perturbations = config.get("local_perturbations", 4)
        self.local_rounds = config.get("local_rounds", 2)
        self.local_epsilon = config.get("local_epsilon", 0.1)

        self.verifiers = config.get("verifiers", ["clip", "image_reward"])
        self.verifier_weights = config.get("verifier_weights", None)

        self.model_paths = config.get("model_paths", {})

        self._teacher_loader: Optional[TeacherLoader] = None
        self._pipeline: Optional[DiffusionPipeline] = None
        self._verifier: Optional[VerifierEnsemble] = None
        self._noise_sampler: Optional[NoiseSampler] = None

    def _get_latent_shape(self) -> tuple:
        """Return latent shape for model family."""
        latent_size = self.image_size // 8
        if self.model_family == "flux":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sd3":
            return (1, 16, latent_size, latent_size)

    def _prepare_latents(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare latents for the pipeline, handling FLUX packing if needed."""
        if isinstance(self._pipeline, FluxPipeline):
            B, C, H, W = noise.shape
            return self._pipeline._pack_latents(noise, B, C, H, W)
        return noise

    def _ensure_loaded(self) -> None:
        """Ensure all components are loaded."""
        if self._pipeline is None:
            self._teacher_loader = TeacherLoader(
                model_family=self.model_family,
                model_path=self.model_paths.get("teacher") if self.model_paths else None,
            )
            self._pipeline = self._teacher_loader.load()

            self._verifier = VerifierEnsemble(
                verifiers=self.verifiers,
                weights=self.verifier_weights,
            )

            self._noise_sampler = NoiseSampler(
                latent_shape=self._get_latent_shape(),
            )

    def _generate_and_score(
        self,
        noise: torch.Tensor,
        prompt: str,
    ) -> tuple:
        """Generate image and score it."""
        latents = self._prepare_latents(noise)
        with torch.no_grad():
            output = self._pipeline(
                prompt=prompt,
                latents=latents,
                num_inference_steps=self.denoising_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                output_type="pil",
            )
        image = output.images[0]
        score = float(self._verifier(image, prompt))
        return image, score

    def __call__(
        self,
        prompt: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full model search.

        Args:
            prompt: Text prompt for generation.
            seed: Random seed.
            verbose: Whether to show progress.

        Returns:
            Dictionary with best image, score, timing, etc.
        """
        self._ensure_loaded()

        timing = {}
        t_total_start = time.time()

        # Phase 1: Broad search
        noises, _ = self._noise_sampler.sample_with_ids(
            num_samples=self.num_candidates,
            base_seed=seed or 0,
            prompt_id=prompt,
        )

        candidates = []
        scores = []

        iterator = range(self.num_candidates)
        if verbose:
            iterator = tqdm(iterator, desc="Full model search")

        t_search_start = time.time()
        for i in iterator:
            image, score = self._generate_and_score(noises[i : i + 1], prompt)
            candidates.append(image)
            scores.append(score)
            torch.cuda.empty_cache()

        timing["broad_search"] = time.time() - t_search_start

        scores = np.array(scores)
        best_idx = int(np.argmax(scores))
        best_noise = noises[best_idx : best_idx + 1].clone()
        best_score = scores[best_idx]
        best_image = candidates[best_idx]

        # Phase 2: Local refinement (optional)
        if self.local_refine:
            t_refine_start = time.time()

            for round_idx in range(self.local_rounds):
                for _ in range(self.local_perturbations):
                    perturbed = self._noise_sampler.perturb(
                        best_noise,
                        epsilon=self.local_epsilon,
                        seed=(seed or 0) + round_idx * 1000 + _,
                    )
                    image, score = self._generate_and_score(perturbed, prompt)

                    if score > best_score:
                        best_noise = perturbed.clone()
                        best_score = score
                        best_image = image

                    torch.cuda.empty_cache()

            timing["local_refinement"] = time.time() - t_refine_start

        timing["total"] = time.time() - t_total_start

        return {
            "image": best_image,
            "score": float(best_score),
            "noise": best_noise,
            "all_scores": scores,
            "timing": timing,
            "config": {
                "method": "full_model_search",
                "num_candidates": self.num_candidates,
                "denoising_steps": self.denoising_steps,
                "local_refine": self.local_refine,
            },
        }

    def unload(self) -> None:
        """Unload models from memory."""
        if self._teacher_loader is not None:
            self._teacher_loader.unload()
        if self._verifier is not None:
            self._verifier.unload()
        torch.cuda.empty_cache()
