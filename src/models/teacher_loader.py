"""Teacher model loader for high-quality refinement."""

from pathlib import Path
from typing import Optional, Union

import torch
from diffusers import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    FluxPipeline,
    LTXPipeline,
    StableDiffusion3Pipeline,
    WanPipeline,
)


class TeacherLoader:
    """
    Loads full teacher models for high-quality image generation.

    Supported teacher models:
    - FLUX-dev: Full 28-step FLUX model (11.9B parameters)
    - SD3.5-Large: Full 28-step SD3.5 model (8.1B parameters)
    """

    SUPPORTED_FAMILIES = ["flux", "sd3", "sdxl", "ltx", "wan"]

    def __init__(
        self,
        model_family: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_path: Optional[Union[str, Path]] = None,
        teacher_lora_path: Optional[str] = None,
    ):
        """
        Initialize the teacher loader.

        Args:
            model_family: Which model family to use ("flux" or "sd3").
            device: Target device for model loading.
            dtype: Model precision (float16 recommended for memory efficiency).
            model_path: Local path to model weights. If None, will attempt to load
                       from default HuggingFace paths (requires network access).
            teacher_lora_path: Optional HF repo id or local path to a LoRA
                       adapter to fuse onto the teacher transformer. Currently
                       wired only for `model_family="sd3"`. PEFT-format
                       (FlowGRPO style) and diffusers-format both supported.

        Raises:
            ValueError: If model_family is not supported.
        """
        if model_family not in self.SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported model family: {model_family}. "
                f"Supported: {self.SUPPORTED_FAMILIES}"
            )

        self.model_family = model_family
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.teacher_lora_path = teacher_lora_path
        self._pipeline: Optional[DiffusionPipeline] = None

    def load(self) -> DiffusionPipeline:
        """
        Load the teacher model pipeline.

        Returns:
            Ready-to-use diffusers pipeline for teacher model.
            - FLUX family: FluxPipeline with dev weights (28-step inference)
            - SD3 family: StableDiffusion3Pipeline with SD3.5-Large weights (28-step inference)

        Note:
            Model weights must be available locally at model_path.
            No automatic downloading is performed.
        """
        if self._pipeline is not None:
            return self._pipeline

        if self.model_family == "flux":
            self._pipeline = self._load_flux_dev()
        elif self.model_family == "sd3":
            self._pipeline = self._load_sd35()
        elif self.model_family == "sdxl":
            self._pipeline = self._load_sdxl()
        elif self.model_family == "ltx":
            self._pipeline = self._load_ltx_dev()
        elif self.model_family == "wan":
            self._pipeline = self._load_wan()

        return self._pipeline

    def _load_flux_dev(self) -> FluxPipeline:
        """Load FLUX-dev pipeline."""
        model_id = self.model_path or "black-forest-labs/FLUX.1-dev"

        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            local_files_only=self.model_path is not None,
        )
        pipeline = pipeline.to(self.device)

        return pipeline

    def _load_sd35(self) -> StableDiffusion3Pipeline:
        """Load SD3.5-Large (or any SD3 base via model_path) pipeline.
        Optionally fuses a LoRA adapter onto the transformer if
        `teacher_lora_path` was provided. Tries diffusers' built-in
        `load_lora_weights` first; falls back to explicit PEFT loading
        for FlowGRPO-style PEFT-format adapters.
        """
        model_id = self.model_path or "stabilityai/stable-diffusion-3.5-large"

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            local_files_only=self.model_path is not None,
        )

        if self.teacher_lora_path:
            lora = str(self.teacher_lora_path)
            # FlowGRPO checkpoints are PEFT-format (prefix `base_model.model.`),
            # which diffusers' load_lora_weights silently no-ops on (it scans for
            # `transformer.` prefix). Use explicit PEFT loading + merge_and_unload.
            from peft import PeftModel
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora)
            pipeline.transformer.set_adapter("default")
            pipeline.transformer = pipeline.transformer.merge_and_unload()
            print(f"[teacher_loader] fused PEFT LoRA from {lora} onto transformer", flush=True)

        pipeline = pipeline.to(self.device)

        return pipeline

    def _load_sdxl(self) -> DiffusionPipeline:
        """Load SDXL-Base 1.0 pipeline. Forces fp16 — SDXL is fp16-native."""
        model_id = self.model_path or "stabilityai/stable-diffusion-xl-base-1.0"

        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=self.model_path is not None,
        )
        pipeline = pipeline.to(self.device)

        return pipeline

    def _load_ltx_dev(self) -> LTXPipeline:
        """Load LTX-Video dev (full) teacher via single-file safetensors.

        The single-file checkpoint only contains the transformer; T5 text encoder,
        tokenizer, VAE, and scheduler must be loaded from the same parent directory.
        """
        from transformers import T5EncoderModel, T5Tokenizer
        from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler

        assert self.model_path is not None, "LTX single-file loader requires model_path"
        repo_root = Path(self.model_path).parent

        text_encoder = T5EncoderModel.from_pretrained(
            str(repo_root / "text_encoder"), torch_dtype=self.dtype, local_files_only=True,
        )
        tokenizer = T5Tokenizer.from_pretrained(
            str(repo_root / "tokenizer"), local_files_only=True, use_fast=False,
        )
        vae = AutoencoderKLLTXVideo.from_pretrained(
            str(repo_root / "vae"), torch_dtype=self.dtype, local_files_only=True,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(repo_root / "scheduler"), local_files_only=True,
        )

        pipeline = LTXPipeline.from_single_file(
            str(self.model_path),
            torch_dtype=self.dtype,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            local_files_only=True,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def _load_wan(self) -> WanPipeline:
        """Load Wan2.1-T2V (1.3B) via diffusers from_pretrained from a local snapshot dir.

        Standard diffusers layout — `WanPipeline.from_pretrained` resolves all
        components (umT5 text encoder, tokenizer, VAE, scheduler, transformer).
        """
        assert self.model_path is not None, "Wan loader requires model_path (snapshot dir)"
        pipeline = WanPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def load_pretrained(self, path: str) -> DiffusionPipeline:
        """
        Load teacher model from a local path.

        Args:
            path: Local path to the model weights.

        Returns:
            Loaded diffusion pipeline.
        """
        self.model_path = path
        return self.load()

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._pipeline is not None

    def get_latent_shape(self, image_size: int = 1024) -> tuple:
        """
        Get the latent tensor shape for this model family.

        Args:
            image_size: Target image size (assumes square images).

        Returns:
            Tuple of (batch_size, channels, height, width) for latent tensors.
        """
        latent_size = image_size // 8  # VAE downsampling factor

        if self.model_family == "flux":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sd3":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sdxl":
            return (1, 4, latent_size, latent_size)

    def get_ltx_latent_shape(self, height: int, width: int, num_frames: int) -> tuple:
        """5D latent shape for LTX-Video: (1, 128, T_latent, H/32, W/32)."""
        return (1, 128, (num_frames - 1) // 8 + 1, height // 32, width // 32)

    def get_wan_latent_shape(self, height: int, width: int, num_frames: int,
                             channels: int = 16, t_compress: int = 4,
                             s_compress: int = 8) -> tuple:
        """5D latent shape for Wan2.1-T2V. Defaults verified at smoke test."""
        return (1, channels, (num_frames - 1) // t_compress + 1,
                height // s_compress, width // s_compress)

    def get_default_num_steps(self) -> int:
        """Get the default number of inference steps for this teacher."""
        if self.model_family == "flux":
            return 50  # BFL model card recommendation
        elif self.model_family == "sd3":
            return 28  # Stability AI model card recommendation
        elif self.model_family == "sdxl":
            return 50  # SDXL-Base model card recommendation
        elif self.model_family == "ltx":
            return 50  # LTX-Video dev diffusers default
        elif self.model_family == "wan":
            return 50  # Wan2.1-T2V default

    def get_default_guidance_scale(self) -> float:
        """Get the default guidance scale for this teacher."""
        if self.model_family == "flux":
            return 3.5
        elif self.model_family == "sd3":
            return 4.5
        elif self.model_family == "sdxl":
            return 7.5
        elif self.model_family == "ltx":
            return 3.0  # LTX-Video dev diffusers default
        elif self.model_family == "wan":
            return 6.0  # Wan2.1-T2V model card recommended (sample_guide_scale)
