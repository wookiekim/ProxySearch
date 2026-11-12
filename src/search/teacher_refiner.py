"""Teacher model refinement for high-quality output."""

import time
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers import DiffusionPipeline, FluxPipeline
from PIL import Image


class TeacherRefiner:
    """
    Refines best candidate with full teacher model.

    Takes the winning noise vector from proxy search and generates
    a high-fidelity output using the full teacher model with more
    denoising steps.
    """

    SUPPORTED_STRATEGIES = ["same_noise", "sdedit", "ddim_inversion"]

    def __init__(
        self,
        teacher_pipeline: DiffusionPipeline,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        refinement_strategy: str = "same_noise",
        sdedit_strength: float = 0.5,
        image_size: int = 1024,
    ):
        """
        Initialize the teacher refiner.

        Args:
            teacher_pipeline: Loaded diffusers pipeline for teacher model
                            (e.g., FLUX-dev or SD3.5-Large).
            num_inference_steps: Number of denoising steps (28 for FLUX/SD3.5).
            guidance_scale: Classifier-free guidance scale.
            refinement_strategy: How to use the proxy result:
                - "same_noise": Run teacher from same initial noise z_T.
                - "sdedit": Add noise to proxy output, denoise with teacher.
                - "ddim_inversion": Invert proxy output, denoise with teacher.
            sdedit_strength: Noise strength for SDEdit strategy (0-1).
            image_size: Target image size.

        Raises:
            ValueError: If refinement_strategy is not supported.
        """
        if refinement_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: {refinement_strategy}. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )

        self.teacher_pipeline = teacher_pipeline
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.refinement_strategy = refinement_strategy
        self.sdedit_strength = sdedit_strength
        self.image_size = image_size

    def refine(
        self,
        noise: torch.Tensor,
        prompt: str,
        proxy_image: Optional[Image.Image] = None,
        return_timing: bool = False,
    ) -> Tuple[Image.Image, Optional[Dict[str, float]]]:
        """
        Generate high-fidelity image with teacher model.

        Args:
            noise: Best noise vector from proxy search (1, C, H, W).
            prompt: Text prompt for generation.
            proxy_image: Proxy output image (required for SDEdit/DDIM strategies).
            return_timing: Whether to return timing information.

        Returns:
            Tuple of:
                - image: High-quality teacher-generated PIL image.
                - timing: Dict with timing info (only if return_timing=True).
        """
        timing = {} if return_timing else None
        t_start = time.time()

        if self.refinement_strategy == "same_noise":
            image = self._refine_same_noise(noise, prompt)
        elif self.refinement_strategy == "sdedit":
            if proxy_image is None:
                raise ValueError("SDEdit strategy requires proxy_image")
            image = self._refine_sdedit(proxy_image, prompt)
        elif self.refinement_strategy == "ddim_inversion":
            if proxy_image is None:
                raise ValueError("DDIM inversion strategy requires proxy_image")
            image = self._refine_ddim_inversion(proxy_image, prompt)

        if return_timing:
            timing["refinement_time"] = time.time() - t_start
            timing["strategy"] = self.refinement_strategy
            return image, timing

        return image, None

    def _prepare_latents(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare latents for the pipeline, handling FLUX packing if needed."""
        if isinstance(self.teacher_pipeline, FluxPipeline):
            B, C, H, W = noise.shape
            return self.teacher_pipeline._pack_latents(noise, B, C, H, W)
        return noise

    def _refine_same_noise(
        self,
        noise: torch.Tensor,
        prompt: str,
    ) -> Image.Image:
        """
        Refine using the same initial noise vector.

        This is the simplest strategy: we trust that the noise vector
        that worked well for the proxy will also work well for the teacher,
        exploiting the latent space alignment between distilled and full models.

        Args:
            noise: Initial noise vector (1, C, H, W).
            prompt: Text prompt for generation.

        Returns:
            Generated PIL image.
        """
        latents = self._prepare_latents(noise)

        with torch.no_grad():
            output = self.teacher_pipeline(
                prompt=prompt,
                latents=latents,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                output_type="pil",
            )

        return output.images[0]

    def _refine_sdedit(
        self,
        proxy_image: Image.Image,
        prompt: str,
    ) -> Image.Image:
        """
        Refine using SDEdit (add noise and denoise).

        This preserves more of the proxy's composition while letting
        the teacher add finer details and improve quality.

        Args:
            proxy_image: Image from proxy model.
            prompt: Text prompt for generation.

        Returns:
            Refined PIL image.
        """
        # Calculate number of steps to skip based on strength
        num_skip_steps = int(self.num_inference_steps * (1 - self.sdedit_strength))
        actual_steps = self.num_inference_steps - num_skip_steps

        with torch.no_grad():
            output = self.teacher_pipeline(
                prompt=prompt,
                image=proxy_image,
                strength=self.sdedit_strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                output_type="pil",
            )

        return output.images[0]

    def _refine_ddim_inversion(
        self,
        proxy_image: Image.Image,
        prompt: str,
    ) -> Image.Image:
        """
        Refine using DDIM inversion.

        Inverts the proxy image to latent space and re-denoises with teacher.
        Most faithful to proxy composition but computationally expensive.

        Args:
            proxy_image: Image from proxy model.
            prompt: Text prompt for generation.

        Returns:
            Refined PIL image.

        Note:
            This is a simplified implementation. Full DDIM inversion
            requires careful handling of the inversion process.
        """
        # Encode image to latent
        with torch.no_grad():
            # Get VAE encoder
            vae = self.teacher_pipeline.vae

            # Preprocess image
            from diffusers.image_processor import VaeImageProcessor
            processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
            image_tensor = processor.preprocess(proxy_image)
            image_tensor = image_tensor.to(vae.device, dtype=vae.dtype)

            # Encode to latent
            latent = vae.encode(image_tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor

            # Run denoising from this latent
            # Note: True DDIM inversion would involve running the scheduler backwards
            # This simplified version just uses the encoded latent directly
            output = self.teacher_pipeline(
                prompt=prompt,
                latents=latent,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                output_type="pil",
            )

        return output.images[0]

    def refine_batch(
        self,
        noises: torch.Tensor,
        prompt: str,
        batch_size: int = 1,
    ) -> list:
        """
        Refine multiple candidates with the teacher model.

        Args:
            noises: Batch of noise vectors (K, C, H, W).
            prompt: Text prompt for generation.
            batch_size: Batch size for processing.

        Returns:
            List of K refined PIL images.
        """
        K = noises.shape[0]
        images = []

        for i in range(0, K, batch_size):
            batch_noises = noises[i : i + batch_size]
            batch_prompts = [prompt] * batch_noises.shape[0]
            batch_latents = self._prepare_latents(batch_noises)

            with torch.no_grad():
                output = self.teacher_pipeline(
                    prompt=batch_prompts,
                    latents=batch_latents,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    height=self.image_size,
                    width=self.image_size,
                    output_type="pil",
                )

            images.extend(output.images)
            torch.cuda.empty_cache()

        return images

    def get_timing_estimate(self) -> Dict[str, Any]:
        """
        Get estimated timing for teacher refinement.

        Returns:
            Dictionary with timing estimates.
        """
        # Rough estimates based on model characteristics
        if self.refinement_strategy == "same_noise":
            time_per_image = self.num_inference_steps * 0.14  # ~0.14s per step
        elif self.refinement_strategy == "sdedit":
            effective_steps = int(self.num_inference_steps * self.sdedit_strength)
            time_per_image = effective_steps * 0.14
        elif self.refinement_strategy == "ddim_inversion":
            # Inversion + denoising
            time_per_image = self.num_inference_steps * 0.14 * 2

        return {
            "strategy": self.refinement_strategy,
            "num_inference_steps": self.num_inference_steps,
            "estimated_time_per_image": time_per_image,
        }
