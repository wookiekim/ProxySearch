"""Noise vector generation for candidate search."""

from typing import Optional, Tuple

import torch

from ..utils.seed import get_generator


class NoiseSampler:
    """
    Generates initial noise vectors for diffusion model search.

    This class handles the sampling of i.i.d. Gaussian noise vectors
    that serve as the starting point for diffusion model inference.
    It also supports perturbation of existing noise vectors for
    local search refinement.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, ...],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the noise sampler.

        Args:
            latent_shape: Shape of latent tensors, e.g., (1, 16, 128, 128) for FLUX
                         or (1, 16, 128, 128) for SD3.5. The batch dimension is
                         typically 1 and will be expanded based on num_samples.
            device: Target device for generated tensors.
            dtype: Data type for generated tensors.
        """
        self.latent_shape = latent_shape
        self.device = device
        self.dtype = dtype

        # Extract shape without batch dimension. Support 4D image latents
        # (1, C, H, W) and 5D video latents (1, C, T, H, W). If first dim is 1,
        # strip it; otherwise treat the full shape as per-sample.
        if len(latent_shape) >= 4 and latent_shape[0] == 1:
            self._single_shape = latent_shape[1:]
        else:
            self._single_shape = latent_shape

    def sample(
        self,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample N i.i.d. Gaussian noise vectors.

        Args:
            num_samples: Number of noise vectors to generate (N).
            seed: Random seed for reproducibility. If None, uses
                 non-deterministic sampling.

        Returns:
            Tensor of shape (N, C, H, W) containing sampled noise vectors,
            where C, H, W match the latent_shape dimensions.
        """
        generator = get_generator(seed, device="cpu")

        # Sample on CPU for reproducibility, then move to device
        noise = torch.randn(
            (num_samples, *self._single_shape),
            generator=generator,
            dtype=self.dtype,
        )

        return noise.to(self.device)

    def sample_single(
        self,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample a single noise vector.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Tensor of shape (1, C, H, W) containing a single noise vector.
        """
        return self.sample(num_samples=1, seed=seed)

    def noise_from_id(
        self,
        base_seed: int,
        prompt_id: str,
        candidate_idx: int,
    ) -> torch.Tensor:
        """
        Get a single deterministic noise vector from (base_seed, prompt_id, candidate_idx).

        Reproducible: same inputs always produce the same noise, regardless of
        how many total candidates are sampled.
        """
        from ..utils.seed import get_candidate_seed
        seed = get_candidate_seed(base_seed, prompt_id, candidate_idx)
        return self.sample_single(seed=seed)

    def perturb(
        self,
        noise: torch.Tensor,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate perturbed versions of noise vectors for local search.

        Uses linear interpolation:
            z_perturbed = (1 - epsilon) * z + epsilon * z_new

        where z_new ~ N(0, I). This maintains the statistical properties
        of the original noise while exploring nearby regions.

        Args:
            noise: Original noise tensor of shape (K, C, H, W).
            epsilon: Interpolation coefficient in [0, 1]. Higher values
                    create larger perturbations.
            seed: Random seed for reproducibility.

        Returns:
            Perturbed noise tensor of the same shape as input.
        """
        generator = get_generator(seed, device="cpu")

        # Sample new noise for perturbation
        perturbation = torch.randn(
            noise.shape,
            generator=generator,
            dtype=self.dtype,
        ).to(self.device)

        # Linear interpolation
        perturbed = (1 - epsilon) * noise + epsilon * perturbation

        return perturbed

    def perturb_batch(
        self,
        noise: torch.Tensor,
        num_perturbations: int,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate multiple perturbations for each input noise vector.

        Args:
            noise: Original noise tensor of shape (K, C, H, W).
            num_perturbations: Number of perturbations per noise vector (M).
            epsilon: Interpolation coefficient.
            seed: Random seed for reproducibility.

        Returns:
            Perturbed noise tensor of shape (K * M, C, H, W).
            The layout is: [noise_0_pert_0, noise_0_pert_1, ..., noise_K_pert_M].
        """
        K = noise.shape[0]
        generator = get_generator(seed, device="cpu")

        # Expand noise to (K, M, C, H, W)
        expanded = noise.unsqueeze(1).expand(-1, num_perturbations, *self._single_shape)

        # Sample perturbations
        perturbations = torch.randn(
            (K, num_perturbations, *self._single_shape),
            generator=generator,
            dtype=self.dtype,
        ).to(self.device)

        # Linear interpolation
        perturbed = (1 - epsilon) * expanded + epsilon * perturbations

        # Reshape to (K * M, C, H, W)
        return perturbed.reshape(-1, *self._single_shape)

    def sample_with_ids(
        self,
        num_samples: int,
        base_seed: int,
        prompt_id: str,
    ) -> Tuple[torch.Tensor, list]:
        """
        Sample noise vectors with deterministic per-candidate seeds.

        This ensures exact reproducibility: given the same (base_seed, prompt_id),
        we always get the same noise vectors.

        Args:
            num_samples: Number of noise vectors to generate.
            base_seed: Base seed for the experiment.
            prompt_id: Unique identifier for the prompt.

        Returns:
            Tuple of:
                - Tensor of shape (N, C, H, W) containing noise vectors
                - List of seeds used for each noise vector
        """
        from ..utils.seed import get_candidate_seed

        seeds = [get_candidate_seed(base_seed, prompt_id, i) for i in range(num_samples)]

        # Sample each noise vector with its deterministic seed
        noises = []
        for seed in seeds:
            noise = self.sample_single(seed=seed)
            noises.append(noise)

        return torch.cat(noises, dim=0), seeds

    def save_noises(
        self,
        noises: torch.Tensor,
        save_path: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Save noise vectors to disk for reproducibility.

        Args:
            noises: Noise tensor to save.
            save_path: Path to save the tensor.
            metadata: Optional metadata dict (seeds, prompt_id, etc.).
        """
        save_dict = {"noises": noises.cpu()}
        if metadata:
            save_dict["metadata"] = metadata
        torch.save(save_dict, save_path)

    def load_noises(
        self,
        load_path: str,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Load noise vectors from disk.

        Args:
            load_path: Path to load from.

        Returns:
            Tuple of (noise tensor, metadata dict or None).
        """
        save_dict = torch.load(load_path, map_location=self.device)
        noises = save_dict["noises"].to(self.device, dtype=self.dtype)
        metadata = save_dict.get("metadata")
        return noises, metadata
