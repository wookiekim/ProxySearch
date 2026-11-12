"""Epsilon-greedy local search around top candidates."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .noise_sampler import NoiseSampler
from .proxy_evaluator import ProxyEvaluator


class LocalRefiner:
    """
    Epsilon-greedy local search around top candidates.

    Performs iterative local search by perturbing top-K noise vectors
    and keeping improvements. This helps refine candidates discovered
    in the broad search phase.
    """

    def __init__(
        self,
        proxy_evaluator: ProxyEvaluator,
        noise_sampler: NoiseSampler,
        num_perturbations: int = 8,
        epsilon: float = 0.1,
        num_rounds: int = 4,
    ):
        """
        Initialize the local refiner.

        Args:
            proxy_evaluator: ProxyEvaluator for generating and scoring candidates.
            noise_sampler: NoiseSampler for generating perturbations.
            num_perturbations: Number of perturbations per candidate per round (M).
            epsilon: Perturbation strength (0 = no change, 1 = full replacement).
            num_rounds: Number of local search rounds (typically one per proxy step).
        """
        self.proxy_evaluator = proxy_evaluator
        self.noise_sampler = noise_sampler
        self.num_perturbations = num_perturbations
        self.epsilon = epsilon
        self.num_rounds = num_rounds

    def refine(
        self,
        noises: torch.Tensor,
        scores: np.ndarray,
        prompt: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray, List[Image.Image]]:
        """
        Perform local refinement via epsilon-greedy perturbation.

        For each round and each candidate:
        1. Generate M perturbations of the current noise
        2. Evaluate all perturbations with the proxy model
        3. If any perturbation improves the score, adopt it

        Args:
            noises: Top-K candidate noise vectors of shape (K, C, H, W).
            scores: Current scores for each candidate of shape (K,).
            prompt: Text prompt for generation.
            seed: Random seed for reproducibility.
            verbose: Whether to show progress bars.

        Returns:
            Tuple of:
                - refined_noises: Improved noise vectors (K, C, H, W).
                - refined_scores: Improved scores (K,).
                - refined_images: List of K refined images.
        """
        K = noises.shape[0]
        current_noises = noises.clone()
        current_scores = scores.copy()
        current_images = [None] * K

        # Track improvement statistics
        total_improvements = 0

        iterator = range(self.num_rounds)
        if verbose:
            iterator = tqdm(iterator, desc="Local refinement rounds")

        for round_idx in iterator:
            round_seed = seed + round_idx if seed is not None else None

            # Process each candidate
            for k in range(K):
                # Generate perturbations for this candidate
                candidate_noise = current_noises[k : k + 1]
                perturbation_seed = (
                    round_seed * K + k if round_seed is not None else None
                )

                perturbed_noises = self.noise_sampler.perturb_batch(
                    candidate_noise,
                    num_perturbations=self.num_perturbations,
                    epsilon=self.epsilon,
                    seed=perturbation_seed,
                )

                # Evaluate perturbations
                perturbed_images, perturbed_scores, _ = self.proxy_evaluator.evaluate(
                    perturbed_noises, prompt
                )

                # Check for improvement
                best_perturbation_idx = np.argmax(perturbed_scores)
                best_perturbation_score = perturbed_scores[best_perturbation_idx]

                if best_perturbation_score > current_scores[k]:
                    # Accept the improvement
                    current_noises[k] = perturbed_noises[best_perturbation_idx]
                    current_scores[k] = best_perturbation_score
                    current_images[k] = perturbed_images[best_perturbation_idx]
                    total_improvements += 1

        # Ensure we have images for candidates that didn't improve
        for k in range(K):
            if current_images[k] is None:
                # Generate final image for this candidate
                img, _ = self.proxy_evaluator.evaluate_single(
                    current_noises[k : k + 1], prompt
                )
                current_images[k] = img

        return current_noises, current_scores, current_images

    def refine_single(
        self,
        noise: torch.Tensor,
        score: float,
        prompt: str,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float, Image.Image]:
        """
        Refine a single candidate.

        Args:
            noise: Single noise vector of shape (1, C, H, W).
            score: Current score for the candidate.
            prompt: Text prompt for generation.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (refined_noise, refined_score, refined_image).
        """
        noises, scores, images = self.refine(
            noise,
            np.array([score]),
            prompt,
            seed=seed,
        )
        return noises[0:1], float(scores[0]), images[0]

    def refine_with_schedule(
        self,
        noises: torch.Tensor,
        scores: np.ndarray,
        prompt: str,
        epsilon_schedule: List[float],
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray, List[Image.Image]]:
        """
        Refine with a custom epsilon schedule.

        Uses different perturbation strengths for each round,
        typically starting with larger perturbations and annealing
        to finer refinements.

        Args:
            noises: Top-K candidate noise vectors of shape (K, C, H, W).
            scores: Current scores for each candidate.
            prompt: Text prompt for generation.
            epsilon_schedule: List of epsilon values, one per round.
            seed: Random seed for reproducibility.
            verbose: Whether to show progress bars.

        Returns:
            Tuple of (refined_noises, refined_scores, refined_images).
        """
        K = noises.shape[0]
        current_noises = noises.clone()
        current_scores = scores.copy()
        current_images = [None] * K

        iterator = enumerate(epsilon_schedule)
        if verbose:
            iterator = tqdm(list(iterator), desc="Scheduled refinement")

        for round_idx, epsilon in iterator:
            round_seed = seed + round_idx if seed is not None else None

            for k in range(K):
                candidate_noise = current_noises[k : k + 1]
                perturbation_seed = (
                    round_seed * K + k if round_seed is not None else None
                )

                perturbed_noises = self.noise_sampler.perturb_batch(
                    candidate_noise,
                    num_perturbations=self.num_perturbations,
                    epsilon=epsilon,  # Use scheduled epsilon
                    seed=perturbation_seed,
                )

                perturbed_images, perturbed_scores, _ = self.proxy_evaluator.evaluate(
                    perturbed_noises, prompt
                )

                best_idx = np.argmax(perturbed_scores)
                if perturbed_scores[best_idx] > current_scores[k]:
                    current_noises[k] = perturbed_noises[best_idx]
                    current_scores[k] = perturbed_scores[best_idx]
                    current_images[k] = perturbed_images[best_idx]

        # Generate images for candidates that didn't improve
        for k in range(K):
            if current_images[k] is None:
                img, _ = self.proxy_evaluator.evaluate_single(
                    current_noises[k : k + 1], prompt
                )
                current_images[k] = img

        return current_noises, current_scores, current_images

    def get_timing_info(
        self,
        noises: torch.Tensor,
        scores: np.ndarray,
        prompt: str,
    ) -> Dict[str, float]:
        """
        Get timing breakdown for local refinement.

        Args:
            noises: Input noises for timing test.
            scores: Input scores for timing test.
            prompt: Text prompt for generation.

        Returns:
            Dictionary with timing information.
        """
        K = noises.shape[0]
        total_evaluations = K * self.num_rounds * self.num_perturbations

        t_start = time.time()
        self.refine(noises, scores, prompt, seed=42, verbose=False)
        t_end = time.time()

        total_time = t_end - t_start

        return {
            "total_time": total_time,
            "num_rounds": self.num_rounds,
            "num_candidates": K,
            "perturbations_per_candidate": self.num_perturbations,
            "total_evaluations": total_evaluations,
            "time_per_evaluation": total_time / total_evaluations,
        }
