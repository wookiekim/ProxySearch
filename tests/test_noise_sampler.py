"""Tests for NoiseSampler."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.noise_sampler import NoiseSampler


class TestNoiseSampler:
    """Tests for the NoiseSampler class."""

    @pytest.fixture
    def sampler(self):
        """Create a noise sampler for testing."""
        latent_shape = (1, 16, 8, 8)  # Small for testing
        return NoiseSampler(
            latent_shape=latent_shape,
            device="cpu",
            dtype=torch.float32,
        )

    def test_sample_shape(self, sampler):
        """Test that sampled noise has correct shape."""
        noises = sampler.sample(num_samples=4, seed=42)
        assert noises.shape == (4, 16, 8, 8)

    def test_sample_reproducibility(self, sampler):
        """Test that same seed produces same noise."""
        noises1 = sampler.sample(num_samples=4, seed=42)
        noises2 = sampler.sample(num_samples=4, seed=42)
        assert torch.allclose(noises1, noises2)

    def test_sample_different_seeds(self, sampler):
        """Test that different seeds produce different noise."""
        noises1 = sampler.sample(num_samples=4, seed=42)
        noises2 = sampler.sample(num_samples=4, seed=43)
        assert not torch.allclose(noises1, noises2)

    def test_sample_single(self, sampler):
        """Test single noise sampling."""
        noise = sampler.sample_single(seed=42)
        assert noise.shape == (1, 16, 8, 8)

    def test_perturb_shape(self, sampler):
        """Test that perturbed noise has correct shape."""
        original = sampler.sample(num_samples=2, seed=42)
        perturbed = sampler.perturb(original, epsilon=0.1, seed=43)
        assert perturbed.shape == original.shape

    def test_perturb_not_identical(self, sampler):
        """Test that perturbation changes the noise."""
        original = sampler.sample(num_samples=2, seed=42)
        perturbed = sampler.perturb(original, epsilon=0.1, seed=43)
        assert not torch.allclose(original, perturbed)

    def test_perturb_epsilon_zero(self, sampler):
        """Test that epsilon=0 returns original noise."""
        original = sampler.sample(num_samples=2, seed=42)
        perturbed = sampler.perturb(original, epsilon=0.0, seed=43)
        assert torch.allclose(original, perturbed)

    def test_perturb_batch_shape(self, sampler):
        """Test batch perturbation shape."""
        original = sampler.sample(num_samples=2, seed=42)
        perturbed = sampler.perturb_batch(
            original, num_perturbations=3, epsilon=0.1, seed=43
        )
        # 2 original * 3 perturbations = 6 total
        assert perturbed.shape == (6, 16, 8, 8)

    def test_sample_statistics(self, sampler):
        """Test that samples have approximately unit Gaussian statistics."""
        noises = sampler.sample(num_samples=1000, seed=42)
        mean = noises.mean().item()
        std = noises.std().item()

        # Should be approximately N(0, 1)
        assert abs(mean) < 0.1, f"Mean too far from 0: {mean}"
        assert abs(std - 1.0) < 0.1, f"Std too far from 1: {std}"

    def test_sample_with_ids(self, sampler):
        """Test sampling with deterministic IDs."""
        noises, seeds = sampler.sample_with_ids(
            num_samples=4,
            base_seed=42,
            prompt_id="test_prompt",
        )
        assert noises.shape == (4, 16, 8, 8)
        assert len(seeds) == 4
        assert len(set(seeds)) == 4  # All seeds should be unique


class TestNoiseSamplerDevices:
    """Test NoiseSampler across devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test noise sampling on CUDA."""
        sampler = NoiseSampler(
            latent_shape=(1, 4, 8, 8),
            device="cuda",
            dtype=torch.float16,
        )
        noises = sampler.sample(num_samples=2, seed=42)
        assert noises.device.type == "cuda"
        assert noises.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
