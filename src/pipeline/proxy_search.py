"""Main ProxySearch pipeline combining all components."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from ..models.proxy_loader import ProxyLoader
from ..models.teacher_loader import TeacherLoader
from ..models.verifier_loader import VerifierEnsemble
from ..search.candidate_selector import CandidateSelector
from ..search.local_refiner import LocalRefiner
from ..search.noise_sampler import NoiseSampler
from ..search.proxy_evaluator import ProxyEvaluator
from ..search.teacher_refiner import TeacherRefiner


class ProxySearchPipeline:
    """
    Main ProxySearch pipeline combining all components.

    This pipeline implements the full ProxySearch algorithm:
    1. Sample N initial noise vectors
    2. Evaluate all candidates with fast proxy model
    3. Select top-K candidates
    4. (Optional) Local refinement via epsilon-greedy perturbation
    5. (Optional) Teacher refinement for highest-quality output
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Initialize the ProxySearch pipeline.

        Args:
            config: Full configuration including:
                - model_family: "flux" or "sd3"
                - num_candidates: N (default 256)
                - top_k: K (default 16)
                - local_refine: bool (default True)
                - teacher_refine: bool (default True)
                - verifiers: List[str]
                - verifier_weights: List[float]
                - proxy_steps: int (default 4)
                - teacher_steps: int (default 28)
                - guidance_scale: float (default 3.5)
                - proxy_guidance_scale: float (default 0.0)
                - image_size: int (default 1024)
                - batch_size: int (default 8)
                - local_perturbations: int (default 8)
                - local_rounds: int (default 4)
                - local_epsilon: float (default 0.1)
                - selection_strategy: str (default "top_k")
                - refinement_strategy: str (default "same_noise")
        """
        self.config = config

        # Extract config values with defaults
        self.model_family = config.get("model_family", "flux")
        self.num_candidates = config.get("num_candidates", 256)
        self.top_k = config.get("top_k", 16)
        self.local_refine = config.get("local_refine", True)
        self.teacher_refine = config.get("teacher_refine", True)
        self.image_size = config.get("image_size", 1024)
        self.batch_size = config.get("batch_size", 8)

        # Verifier configuration
        self.verifiers = config.get("verifiers", ["clip", "image_reward"])
        self.verifier_weights = config.get("verifier_weights", None)

        # Proxy configuration
        self.proxy_steps = config.get("proxy_steps", 4)
        self.proxy_guidance_scale = config.get("proxy_guidance_scale", 0.0)

        # Teacher configuration
        self.teacher_steps = config.get("teacher_steps", 28)
        self.teacher_guidance_scale = config.get("guidance_scale", 3.5)
        self.refinement_strategy = config.get("refinement_strategy", "same_noise")

        # Local refinement configuration
        self.local_perturbations = config.get("local_perturbations", 8)
        self.local_rounds = config.get("local_rounds", 4)
        self.local_epsilon = config.get("local_epsilon", 0.1)

        # Selection configuration
        self.selection_strategy = config.get("selection_strategy", "top_k")

        # Model paths (optional)
        self.model_paths = config.get("model_paths", {})

        # Initialize components (lazy loading)
        self._proxy_loader: Optional[ProxyLoader] = None
        self._teacher_loader: Optional[TeacherLoader] = None
        self._verifier: Optional[VerifierEnsemble] = None
        self._noise_sampler: Optional[NoiseSampler] = None
        self._proxy_evaluator: Optional[ProxyEvaluator] = None
        self._selector: Optional[CandidateSelector] = None
        self._local_refiner: Optional[LocalRefiner] = None
        self._teacher_refiner: Optional[TeacherRefiner] = None

    def _get_latent_shape(self) -> Tuple[int, ...]:
        """Return latent shape for model family."""
        latent_size = self.image_size // 8
        if self.model_family == "flux":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sd3":
            return (1, 16, latent_size, latent_size)
        else:
            raise ValueError(f"Unknown model family: {self.model_family}")

    def _ensure_components_loaded(self) -> None:
        """Ensure all components are loaded."""
        if self._proxy_loader is None:
            self._load_components()

    def _load_components(self) -> None:
        """Load all pipeline components."""
        # Load proxy model
        self._proxy_loader = ProxyLoader(
            model_family=self.model_family,
            model_path=self.model_paths.get("proxy"),
            lcm_lora_path=self.model_paths.get("lcm_lora"),
        )
        proxy_pipeline = self._proxy_loader.load()

        # Load teacher model (if needed)
        if self.teacher_refine:
            self._teacher_loader = TeacherLoader(
                model_family=self.model_family,
                model_path=self.model_paths.get("teacher"),
            )
            teacher_pipeline = self._teacher_loader.load()
        else:
            teacher_pipeline = None

        # Load verifiers
        self._verifier = VerifierEnsemble(
            verifiers=self.verifiers,
            weights=self.verifier_weights,
            model_paths=self.model_paths.get("verifiers", {}),
        )

        # Initialize search components
        self._noise_sampler = NoiseSampler(
            latent_shape=self._get_latent_shape(),
        )

        self._proxy_evaluator = ProxyEvaluator(
            proxy_pipeline=proxy_pipeline,
            verifier=self._verifier,
            num_inference_steps=self.proxy_steps,
            guidance_scale=self.proxy_guidance_scale,
            batch_size=self.batch_size,
            image_size=self.image_size,
        )

        self._selector = CandidateSelector(
            top_k=self.top_k,
            selection_strategy=self.selection_strategy,
        )

        # Initialize optional refiners
        if self.local_refine:
            self._local_refiner = LocalRefiner(
                proxy_evaluator=self._proxy_evaluator,
                noise_sampler=self._noise_sampler,
                num_perturbations=self.local_perturbations,
                epsilon=self.local_epsilon,
                num_rounds=self.local_rounds,
            )

        if self.teacher_refine and teacher_pipeline is not None:
            self._teacher_refiner = TeacherRefiner(
                teacher_pipeline=teacher_pipeline,
                num_inference_steps=self.teacher_steps,
                guidance_scale=self.teacher_guidance_scale,
                refinement_strategy=self.refinement_strategy,
                image_size=self.image_size,
            )

    def __call__(
        self,
        prompt: str,
        seed: Optional[int] = None,
        return_all_candidates: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full ProxySearch pipeline.

        Args:
            prompt: Text prompt for generation.
            seed: Random seed for reproducibility.
            return_all_candidates: Whether to return all N candidates.
            verbose: Whether to show progress information.

        Returns:
            Dictionary with keys:
                - "image": Final output PIL image.
                - "score": Final verifier score.
                - "noise": Winning noise vector.
                - "timing": Dict of phase timings.
                - "statistics": Dict of score statistics.
                - "candidates": (optional) All candidate images/scores.
        """
        self._ensure_components_loaded()

        timing = {}
        result = {}

        # Phase 1: Sample initial noise vectors
        t_start = time.time()
        noises, noise_seeds = self._noise_sampler.sample_with_ids(
            num_samples=self.num_candidates,
            base_seed=seed or 0,
            prompt_id=prompt,
        )
        timing["noise_sampling"] = time.time() - t_start

        # Phase 1a: Proxy evaluation
        t_start = time.time()
        if verbose:
            images, scores = self._proxy_evaluator.evaluate_with_progress(
                noises, prompt, desc="Proxy evaluation"
            )
        else:
            images, scores, _ = self._proxy_evaluator.evaluate(noises, prompt)
        timing["proxy_evaluation"] = time.time() - t_start

        # Store statistics before selection
        result["statistics"] = self._proxy_evaluator.compute_statistics(scores)

        # Return all candidates if requested
        if return_all_candidates:
            result["all_candidates"] = {
                "images": images,
                "scores": scores.copy(),
                "noises": noises.clone(),
            }

        # Phase 1b: Select top-K candidates
        t_start = time.time()
        selected_scores, selected_images, selected_noises, selected_indices = (
            self._selector.select(scores, images, noises)
        )
        timing["selection"] = time.time() - t_start

        # Phase 1c: Local refinement (optional)
        if self.local_refine and self._local_refiner is not None:
            t_start = time.time()
            refined_noises, refined_scores, refined_images = self._local_refiner.refine(
                selected_noises,
                selected_scores,
                prompt,
                seed=seed,
                verbose=verbose,
            )
            timing["local_refinement"] = time.time() - t_start

            # Update with refined results
            selected_noises = refined_noises
            selected_scores = refined_scores
            selected_images = refined_images

        # Select best candidate after refinement
        best_idx = int(np.argmax(selected_scores))
        best_noise = selected_noises[best_idx : best_idx + 1]
        best_score = float(selected_scores[best_idx])
        best_proxy_image = selected_images[best_idx]

        # Phase 2: Teacher refinement (optional)
        if self.teacher_refine and self._teacher_refiner is not None:
            t_start = time.time()
            final_image, _ = self._teacher_refiner.refine(
                best_noise,
                prompt,
                proxy_image=best_proxy_image,
            )
            timing["teacher_refinement"] = time.time() - t_start

            # Re-score the teacher output
            final_score = float(self._verifier(final_image, prompt))
        else:
            final_image = best_proxy_image
            final_score = best_score

        # Compute total time
        timing["total"] = sum(timing.values())

        # Assemble result
        result["image"] = final_image
        result["score"] = final_score
        result["proxy_score"] = best_score
        result["noise"] = best_noise
        result["timing"] = timing
        result["config"] = {
            "num_candidates": self.num_candidates,
            "top_k": self.top_k,
            "local_refine": self.local_refine,
            "teacher_refine": self.teacher_refine,
        }

        return result

    def generate_without_search(
        self,
        prompt: str,
        seed: Optional[int] = None,
        use_teacher: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a single image without search (for comparison).

        Args:
            prompt: Text prompt for generation.
            seed: Random seed.
            use_teacher: Whether to use teacher (True) or proxy (False).

        Returns:
            Dictionary with image, score, and timing.
        """
        self._ensure_components_loaded()

        timing = {}
        t_start = time.time()

        # Sample single noise
        noise = self._noise_sampler.sample_single(seed=seed)

        if use_teacher and self._teacher_refiner is not None:
            image, _ = self._teacher_refiner.refine(noise, prompt)
        else:
            image, _ = self._proxy_evaluator.evaluate_single(noise, prompt)

        timing["generation"] = time.time() - t_start

        score = float(self._verifier(image, prompt))

        return {
            "image": image,
            "score": score,
            "noise": noise,
            "timing": timing,
        }

    def unload(self) -> None:
        """Unload all models from memory."""
        if self._proxy_loader is not None:
            self._proxy_loader.unload()
        if self._teacher_loader is not None:
            self._teacher_loader.unload()
        if self._verifier is not None:
            self._verifier.unload()

        self._proxy_loader = None
        self._teacher_loader = None
        self._verifier = None
        self._noise_sampler = None
        self._proxy_evaluator = None
        self._selector = None
        self._local_refiner = None
        self._teacher_refiner = None

        torch.cuda.empty_cache()

    @classmethod
    def from_config_path(cls, config_path: str) -> "ProxySearchPipeline":
        """
        Create pipeline from a config file path.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Initialized ProxySearchPipeline.
        """
        from ..utils.config import load_config

        config = load_config(config_path)
        return cls(config)
