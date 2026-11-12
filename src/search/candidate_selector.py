"""Candidate selection based on verifier scores."""

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


class CandidateSelector:
    """
    Selects top-K candidates based on verifier scores.

    Supports multiple selection strategies for exploring
    different trade-offs between exploitation and exploration.
    """

    SUPPORTED_STRATEGIES = ["top_k", "tournament", "softmax", "epsilon_greedy"]

    def __init__(
        self,
        top_k: int = 16,
        selection_strategy: str = "top_k",
        temperature: float = 1.0,
        epsilon: float = 0.1,
    ):
        """
        Initialize the candidate selector.

        Args:
            top_k: Number of candidates to select (K).
            selection_strategy: Selection method:
                - "top_k": Deterministically select top-K by score.
                - "tournament": Tournament selection with replacement.
                - "softmax": Sample K candidates with softmax probabilities.
                - "epsilon_greedy": Mix of top-K and random selection.
            temperature: Temperature for softmax sampling (lower = more greedy).
            epsilon: Probability of random selection in epsilon_greedy strategy.

        Raises:
            ValueError: If strategy is not supported.
        """
        if selection_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: {selection_strategy}. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )

        self.top_k = top_k
        self.selection_strategy = selection_strategy
        self.temperature = temperature
        self.epsilon = epsilon

    def select(
        self,
        scores: np.ndarray,
        images: List[Image.Image],
        noises: torch.Tensor,
    ) -> Tuple[np.ndarray, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Select top-K candidates based on scores.

        Args:
            scores: Array of N verifier scores.
            images: List of N PIL images.
            noises: Tensor of shape (N, C, H, W) noise vectors.

        Returns:
            Tuple of:
                - selected_scores: (K,) array of selected scores.
                - selected_images: List of K selected images.
                - selected_noises: (K, C, H, W) tensor of selected noises.
                - selected_indices: (K,) array of original indices.
        """
        if self.selection_strategy == "top_k":
            indices = self._select_top_k(scores)
        elif self.selection_strategy == "tournament":
            indices = self._select_tournament(scores)
        elif self.selection_strategy == "softmax":
            indices = self._select_softmax(scores)
        elif self.selection_strategy == "epsilon_greedy":
            indices = self._select_epsilon_greedy(scores)

        # Gather selected candidates
        selected_scores = scores[indices]
        selected_images = [images[i] for i in indices]
        selected_noises = noises[indices]

        return selected_scores, selected_images, selected_noises, indices

    def _select_top_k(self, scores: np.ndarray) -> np.ndarray:
        """Deterministically select top-K by score."""
        k = min(self.top_k, len(scores))
        # Use .copy() to avoid negative stride issues with PyTorch indexing
        indices = np.argsort(scores)[-k:][::-1].copy()  # Descending order
        return indices

    def _select_tournament(self, scores: np.ndarray) -> np.ndarray:
        """
        Tournament selection with replacement.

        For each slot, randomly select 2 candidates and keep the better one.
        """
        k = min(self.top_k, len(scores))
        n = len(scores)
        selected = []

        for _ in range(k):
            # Random tournament of size 2
            idx1, idx2 = np.random.choice(n, size=2, replace=False)
            winner = idx1 if scores[idx1] >= scores[idx2] else idx2
            selected.append(winner)

        return np.array(selected)

    def _select_softmax(self, scores: np.ndarray) -> np.ndarray:
        """Sample K candidates using softmax probabilities."""
        k = min(self.top_k, len(scores))

        # Compute softmax probabilities
        exp_scores = np.exp((scores - np.max(scores)) / self.temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Sample without replacement
        indices = np.random.choice(
            len(scores),
            size=k,
            replace=False,
            p=probs,
        )

        return indices

    def _select_epsilon_greedy(self, scores: np.ndarray) -> np.ndarray:
        """
        Mix of greedy top-K and random selection.

        With probability (1-epsilon), select top-K.
        With probability epsilon, select randomly.
        """
        k = min(self.top_k, len(scores))
        n = len(scores)

        # Determine how many to select randomly
        num_random = int(np.round(k * self.epsilon))
        num_greedy = k - num_random

        # Get top candidates (use .copy() to avoid negative stride issues)
        top_indices = np.argsort(scores)[-num_greedy:][::-1].copy().tolist()

        # Get random candidates (excluding already selected)
        remaining = set(range(n)) - set(top_indices)
        random_indices = np.random.choice(
            list(remaining),
            size=min(num_random, len(remaining)),
            replace=False,
        ).tolist()

        indices = np.array(top_indices + random_indices)
        return indices

    def select_best(
        self,
        scores: np.ndarray,
        images: List[Image.Image],
        noises: torch.Tensor,
    ) -> Tuple[float, Image.Image, torch.Tensor, int]:
        """
        Select the single best candidate.

        Args:
            scores: Array of N verifier scores.
            images: List of N PIL images.
            noises: Tensor of shape (N, C, H, W) noise vectors.

        Returns:
            Tuple of (best_score, best_image, best_noise, best_index).
        """
        best_idx = int(np.argmax(scores))
        return (
            float(scores[best_idx]),
            images[best_idx],
            noises[best_idx : best_idx + 1],
            best_idx,
        )

    def select_diverse(
        self,
        scores: np.ndarray,
        images: List[Image.Image],
        noises: torch.Tensor,
        diversity_weight: float = 0.3,
    ) -> Tuple[np.ndarray, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Select candidates balancing score and diversity.

        Uses a greedy algorithm that iteratively selects candidates
        that maximize a weighted combination of score and distance
        from already selected candidates (in noise space).

        Args:
            scores: Array of N verifier scores.
            images: List of N PIL images.
            noises: Tensor of shape (N, C, H, W) noise vectors.
            diversity_weight: Weight for diversity term (0 = pure score).

        Returns:
            Tuple of (selected_scores, selected_images, selected_noises, selected_indices).
        """
        k = min(self.top_k, len(scores))
        n = len(scores)

        # Normalize scores to [0, 1]
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Flatten noises for distance computation
        noises_flat = noises.reshape(n, -1).cpu().numpy()

        selected_indices = []
        remaining = set(range(n))

        for i in range(k):
            best_score = float("-inf")
            best_idx = None

            for idx in remaining:
                # Score component
                score_value = normalized_scores[idx]

                # Diversity component (min distance to selected)
                if selected_indices:
                    selected_noises_flat = noises_flat[selected_indices]
                    distances = np.linalg.norm(
                        selected_noises_flat - noises_flat[idx], axis=1
                    )
                    diversity_value = np.min(distances)
                    # Normalize diversity
                    diversity_value = diversity_value / (
                        np.sqrt(noises_flat.shape[1]) + 1e-8
                    )
                else:
                    diversity_value = 1.0

                # Combined score
                combined = (1 - diversity_weight) * score_value + diversity_weight * diversity_value

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        indices = np.array(selected_indices)
        selected_scores = scores[indices]
        selected_images = [images[i] for i in indices]
        selected_noises = noises[indices]

        return selected_scores, selected_images, selected_noises, indices
