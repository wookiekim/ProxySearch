"""Tests for ProxySearch pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineConfig:
    """Test pipeline configuration handling."""

    def test_default_config_values(self):
        """Test that default config values are applied."""
        config = OmegaConf.create({
            "model_family": "flux",
        })

        from src.pipeline.proxy_search import ProxySearchPipeline

        # Just test that the class can be instantiated with minimal config
        pipeline = ProxySearchPipeline(config)

        assert pipeline.model_family == "flux"
        assert pipeline.num_candidates == 256  # default
        assert pipeline.top_k == 16  # default

    def test_config_override(self):
        """Test that config values can be overridden."""
        config = OmegaConf.create({
            "model_family": "sd3",
            "num_candidates": 128,
            "top_k": 8,
            "local_refine": False,
            "teacher_refine": False,
        })

        from src.pipeline.proxy_search import ProxySearchPipeline

        pipeline = ProxySearchPipeline(config)

        assert pipeline.model_family == "sd3"
        assert pipeline.num_candidates == 128
        assert pipeline.top_k == 8
        assert pipeline.local_refine is False

    def test_latent_shape_flux(self):
        """Test FLUX latent shape computation."""
        config = OmegaConf.create({
            "model_family": "flux",
            "image_size": 1024,
        })

        from src.pipeline.proxy_search import ProxySearchPipeline

        pipeline = ProxySearchPipeline(config)
        shape = pipeline._get_latent_shape()

        assert shape == (1, 16, 128, 128)

    def test_latent_shape_sd3(self):
        """Test SD3.5 latent shape computation."""
        config = OmegaConf.create({
            "model_family": "sd3",
            "image_size": 1024,
        })

        from src.pipeline.proxy_search import ProxySearchPipeline

        pipeline = ProxySearchPipeline(config)
        shape = pipeline._get_latent_shape()

        assert shape == (1, 16, 128, 128)


class TestBaselineConfig:
    """Test baseline configuration handling."""

    def test_best_of_n_config(self):
        """Test BestOfN configuration."""
        config = OmegaConf.create({
            "model_family": "flux",
            "num_candidates": 16,
            "denoising_steps": 28,
            "guidance_scale": 3.5,
        })

        from src.pipeline.baselines import BestOfN

        baseline = BestOfN(config)

        assert baseline.num_candidates == 16
        assert baseline.denoising_steps == 28

    def test_random_search_config(self):
        """Test RandomSearch configuration."""
        config = OmegaConf.create({
            "model_family": "flux",
            "denoising_steps": 28,
        })

        from src.pipeline.baselines import RandomSearch

        baseline = RandomSearch(config)

        assert baseline.denoising_steps == 28


class TestPipelineComponents:
    """Test individual pipeline components work together."""

    def test_noise_sampler_to_selector(self):
        """Test noise flows from sampler to selector."""
        from src.search.noise_sampler import NoiseSampler
        from src.search.candidate_selector import CandidateSelector

        # Create sampler
        sampler = NoiseSampler(
            latent_shape=(1, 4, 8, 8),
            device="cpu",
            dtype=torch.float32,
        )

        # Sample noises
        noises = sampler.sample(num_samples=8, seed=42)

        # Create mock images and scores
        images = [Image.new("RGB", (64, 64)) for _ in range(8)]
        scores = np.random.random(8)

        # Select candidates
        selector = CandidateSelector(top_k=4)
        selected_scores, selected_images, selected_noises, indices = selector.select(
            scores, images, noises
        )

        assert selected_noises.shape[0] == 4
        assert len(selected_images) == 4

    def test_config_loading(self):
        """Test config loading utility."""
        from src.utils.config import load_config, merge_configs

        # Create temporary config
        config = OmegaConf.create({
            "model_family": "flux",
            "num_candidates": 256,
        })

        override = OmegaConf.create({
            "num_candidates": 128,
        })

        merged = merge_configs(config, [override])

        assert merged.num_candidates == 128
        assert merged.model_family == "flux"


class TestPipelineFromConfigPath:
    """Test pipeline creation from config path."""

    def test_from_config_dict(self):
        """Test creating pipeline from config dict."""
        config = OmegaConf.create({
            "model_family": "flux",
            "num_candidates": 64,
            "top_k": 8,
            "local_refine": False,
            "teacher_refine": False,
            "proxy_steps": 4,
            "teacher_steps": 28,
            "guidance_scale": 3.5,
            "proxy_guidance_scale": 0.0,
            "image_size": 1024,
            "batch_size": 8,
            "verifiers": ["clip", "image_reward"],
            "selection_strategy": "top_k",
            "refinement_strategy": "same_noise",
        })

        from src.pipeline.proxy_search import ProxySearchPipeline

        pipeline = ProxySearchPipeline(config)

        assert pipeline.config is not None
        assert pipeline.num_candidates == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
