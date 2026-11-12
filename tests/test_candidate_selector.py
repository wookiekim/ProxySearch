"""Tests for CandidateSelector."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.candidate_selector import CandidateSelector


class TestCandidateSelector:
    """Tests for the CandidateSelector class."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        np.random.seed(42)
        scores = np.array([0.1, 0.5, 0.3, 0.8, 0.6, 0.2, 0.9, 0.4])
        images = [
            Image.new("RGB", (64, 64), color=(i * 30, i * 30, i * 30))
            for i in range(8)
        ]
        noises = torch.randn(8, 4, 8, 8)
        return scores, images, noises

    def test_top_k_selection(self, mock_data):
        """Test top-K selection strategy."""
        scores, images, noises = mock_data
        selector = CandidateSelector(top_k=3, selection_strategy="top_k")

        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        # Should select indices 6, 3, 4 (scores 0.9, 0.8, 0.6)
        assert len(indices) == 3
        assert 6 in indices  # 0.9
        assert 3 in indices  # 0.8
        assert selected_scores[0] == 0.9  # Best score first

    def test_top_k_larger_than_candidates(self, mock_data):
        """Test when top_k is larger than number of candidates."""
        scores, images, noises = mock_data
        selector = CandidateSelector(top_k=100, selection_strategy="top_k")

        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        # Should return all candidates
        assert len(indices) == len(scores)

    def test_select_best(self, mock_data):
        """Test selecting single best candidate."""
        scores, images, noises = mock_data
        selector = CandidateSelector(top_k=1)

        best_score, best_image, best_noise, best_idx = selector.select_best(
            scores, images, noises
        )

        assert best_idx == 6  # Index of 0.9
        assert best_score == 0.9
        assert best_noise.shape == (1, 4, 8, 8)

    def test_tournament_selection(self, mock_data):
        """Test tournament selection strategy."""
        scores, images, noises = mock_data
        selector = CandidateSelector(top_k=3, selection_strategy="tournament")

        np.random.seed(42)  # For reproducibility
        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert len(indices) == 3
        # Tournament selection should favor higher scores
        assert np.mean(selected_scores) > np.mean(scores)

    def test_softmax_selection(self, mock_data):
        """Test softmax selection strategy."""
        scores, images, noises = mock_data
        selector = CandidateSelector(
            top_k=3, selection_strategy="softmax", temperature=0.5
        )

        np.random.seed(42)
        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert len(indices) == 3
        # All indices should be unique (sampling without replacement)
        assert len(set(indices)) == 3

    def test_epsilon_greedy_selection(self, mock_data):
        """Test epsilon-greedy selection strategy."""
        scores, images, noises = mock_data
        selector = CandidateSelector(
            top_k=4, selection_strategy="epsilon_greedy", epsilon=0.5
        )

        np.random.seed(42)
        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert len(indices) == 4
        # With 50% epsilon, should have some greedy and some random

    def test_select_diverse(self, mock_data):
        """Test diversity-aware selection."""
        scores, images, noises = mock_data
        selector = CandidateSelector(top_k=3)

        selected_scores, selected_images, selected_noises, indices = selector.select_diverse(
            scores, images, noises, diversity_weight=0.5
        )

        assert len(indices) == 3
        # Should balance score and diversity

    def test_output_types(self, mock_data):
        """Test that outputs have correct types."""
        scores, images, noises = mock_data
        selector = CandidateSelector(top_k=2)

        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert isinstance(selected_scores, np.ndarray)
        assert isinstance(selected_images, list)
        assert all(isinstance(img, Image.Image) for img in selected_images)
        assert isinstance(selected_noises, torch.Tensor)
        assert isinstance(indices, np.ndarray)


class TestCandidateSelectorEdgeCases:
    """Test edge cases for CandidateSelector."""

    def test_single_candidate(self):
        """Test with single candidate."""
        selector = CandidateSelector(top_k=1)
        scores = np.array([0.5])
        images = [Image.new("RGB", (64, 64))]
        noises = torch.randn(1, 4, 8, 8)

        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert len(indices) == 1
        assert indices[0] == 0

    def test_identical_scores(self):
        """Test with identical scores."""
        selector = CandidateSelector(top_k=2)
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        images = [Image.new("RGB", (64, 64)) for _ in range(4)]
        noises = torch.randn(4, 4, 8, 8)

        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert len(indices) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
