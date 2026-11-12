"""Tests for ProxyEvaluator (unit tests without models)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockVerifier:
    """Mock verifier for testing."""

    def __call__(self, images, prompt):
        """Return mock scores."""
        if isinstance(images, list):
            # Return deterministic scores based on image index
            return np.array([0.5 + 0.1 * i for i in range(len(images))])
        return 0.5


class MockPipeline:
    """Mock diffusion pipeline for testing."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt, latents, num_inference_steps, guidance_scale,
                 height, width, output_type):
        """Return mock images."""
        batch_size = latents.shape[0]
        self.call_count += 1

        class MockOutput:
            def __init__(self, n):
                self.images = [
                    Image.new("RGB", (64, 64), color=(i * 30, i * 30, i * 30))
                    for i in range(n)
                ]

        return MockOutput(batch_size)


class TestProxyEvaluatorUnit:
    """Unit tests for ProxyEvaluator using mocks."""

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        from src.search.proxy_evaluator import ProxyEvaluator

        return ProxyEvaluator(
            proxy_pipeline=MockPipeline(),
            verifier=MockVerifier(),
            num_inference_steps=4,
            guidance_scale=0.0,
            batch_size=2,
            image_size=64,
        )

    def test_evaluate_returns_correct_count(self, mock_evaluator):
        """Test that evaluate returns correct number of images and scores."""
        noises = torch.randn(4, 16, 8, 8)
        prompt = "test prompt"

        images, scores, _ = mock_evaluator.evaluate(noises, prompt)

        assert len(images) == 4
        assert len(scores) == 4

    def test_evaluate_batches_correctly(self, mock_evaluator):
        """Test that evaluation batches correctly."""
        noises = torch.randn(5, 16, 8, 8)  # 5 samples, batch_size=2
        prompt = "test prompt"

        images, scores, _ = mock_evaluator.evaluate(noises, prompt)

        # Should have processed 3 batches (2, 2, 1)
        assert mock_evaluator.proxy_pipeline.call_count == 3
        assert len(images) == 5

    def test_evaluate_single(self, mock_evaluator):
        """Test single noise evaluation."""
        noise = torch.randn(1, 16, 8, 8)
        prompt = "test prompt"

        image, score = mock_evaluator.evaluate_single(noise, prompt)

        assert isinstance(image, Image.Image)
        assert isinstance(score, float)

    def test_compute_statistics(self, mock_evaluator):
        """Test statistics computation."""
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        stats = mock_evaluator.compute_statistics(scores)

        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "min" in stats
        assert "median" in stats

        assert abs(stats["mean"] - 0.7) < 0.01
        assert stats["max"] == 0.9
        assert stats["min"] == 0.5

    def test_timing_returned(self, mock_evaluator):
        """Test that timing info is returned when requested."""
        noises = torch.randn(2, 16, 8, 8)
        prompt = "test prompt"

        images, scores, timing = mock_evaluator.evaluate(
            noises, prompt, return_timing=True
        )

        assert timing is not None
        assert "generation" in timing
        assert "scoring" in timing


class TestProxyEvaluatorIntegration:
    """Integration tests (require actual models - skip if not available)."""

    @pytest.mark.skip(reason="Requires GPU and models")
    def test_real_evaluation(self):
        """Test with real models (skipped by default)."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
