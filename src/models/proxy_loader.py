"""Proxy model loader for fast candidate evaluation."""

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


class ProxyLoader:
    """
    Loads distilled proxy models for fast candidate evaluation.

    Supported proxy models:
    - FLUX-schnell: 4-step distilled FLUX model (DiT, 11.9B)
    - SD3.5-Large-Turbo: 4-step distilled SD3.5 model (MMDiT, 8.1B)
    """

    SUPPORTED_FAMILIES = ["flux", "sd3", "sdxl", "ltx", "wan"]

    def __init__(
        self,
        model_family: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_path: Optional[Union[str, Path]] = None,
        lcm_lora_path: Optional[Union[str, Path]] = None,
        proxy_unet_override_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the proxy loader.

        Args:
            model_family: Which proxy-teacher pair to use ("flux" or "sd3").
            device: Target device for model loading.
            dtype: Model precision (float16 recommended for memory efficiency).
            model_path: Local path to model weights. If None, will attempt to load
                       from default HuggingFace paths (requires network access).
            lcm_lora_path: SDXL-only. If set, load SDXL-Base teacher and apply this
                           LCM-LoRA adapter (with LCMScheduler) instead of SDXL-Turbo.
            proxy_unet_override_path: SDXL-only. If set, load SDXL-Base and replace
                                       the UNet weights with this checkpoint (for
                                       DMD2 / Hyper-SD / SDXL-Lightning style full
                                       UNet swaps). Use prefix "hf:<repo>/<file>"
                                       to fetch from HuggingFace, else absolute path.

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
        self.lcm_lora_path = lcm_lora_path
        self.proxy_unet_override_path = proxy_unet_override_path
        self._pipeline: Optional[DiffusionPipeline] = None

    def load(self) -> DiffusionPipeline:
        """
        Load the proxy model pipeline.

        Returns:
            Ready-to-use diffusers pipeline for proxy model.
            - FLUX family: FluxPipeline with schnell weights (4-step inference)
            - SD3 family: StableDiffusion3Pipeline with SD3.5-Large-Turbo (4-step inference)

        Note:
            Model weights must be available locally at model_path.
            No automatic downloading is performed.
        """
        if self._pipeline is not None:
            return self._pipeline

        if self.model_family == "flux":
            self._pipeline = self._load_flux_schnell()
        elif self.model_family == "sd3":
            self._pipeline = self._load_sd35_turbo()
        elif self.model_family == "sdxl":
            if self.lcm_lora_path:
                self._pipeline = self._load_sdxl_lcm()
            elif self.proxy_unet_override_path:
                self._pipeline = self._load_sdxl_unet_override()
            else:
                self._pipeline = self._load_sdxl_turbo()
        elif self.model_family == "ltx":
            self._pipeline = self._load_ltx_distilled()
        elif self.model_family == "wan":
            self._pipeline = self._load_wan()

        return self._pipeline

    def _load_flux_schnell(self) -> FluxPipeline:
        """Load FLUX-schnell pipeline."""
        # Default path for FLUX-schnell if not specified
        model_id = self.model_path or "black-forest-labs/FLUX.1-schnell"

        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            local_files_only=self.model_path is not None,
        )
        pipeline = pipeline.to(self.device)

        return pipeline

    def _load_sd35_turbo(self) -> StableDiffusion3Pipeline:
        """Load SD3.5-Large-Turbo pipeline."""
        model_id = self.model_path or "stabilityai/stable-diffusion-3.5-large-turbo"

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            local_files_only=self.model_path is not None,
        )
        pipeline = pipeline.to(self.device)

        return pipeline

    def _load_sdxl_turbo(self) -> DiffusionPipeline:
        """Load SDXL-Turbo pipeline. Forces fp16 — SDXL is fp16-native."""
        model_id = self.model_path or "stabilityai/sdxl-turbo"

        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=self.model_path is not None,
        )
        pipeline = pipeline.to(self.device)

        return pipeline

    def _load_sdxl_lcm(self) -> DiffusionPipeline:
        """Load SDXL-Base + LCM-LoRA adapter; swap scheduler to LCMScheduler.

        Consistency-distillation proxy variant: SDXL-Base UNet with the
        LCM-LoRA adapter fused in, paired with `LCMScheduler` for 4-step
        rendering at low CFG. Same SDXL-Base teacher is used for the
        50-step reference (loaded separately by TeacherLoader).
        """
        from diffusers import LCMScheduler

        base_id = self.model_path or "stabilityai/stable-diffusion-xl-base-1.0"
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=self.model_path is not None,
        )
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights(str(self.lcm_lora_path))
        pipeline.fuse_lora()
        pipeline = pipeline.to(self.device)
        return pipeline

    def _load_sdxl_unet_override(self) -> DiffusionPipeline:
        """Load SDXL-Base, replace UNet weights with a distilled-sibling UNet.

        Score-distillation / progressive-adversarial proxy variant: load
        SDXL-Base, then `load_state_dict` the override UNet (e.g. DMD2
        4-step UNet, Hyper-SD UNet, SDXL-Lightning UNet). The text encoders,
        VAE, and tokenizers stay at SDXL-Base values; only the UNet is
        swapped. Same SDXL-Base teacher is used for the 50-step reference
        (loaded separately by TeacherLoader).

        `proxy_unet_override_path` accepts:
          - "hf:<repo_id>/<filename>" → download via huggingface_hub
          - absolute filesystem path → load from disk
        """
        base_id = self.model_path or "stabilityai/stable-diffusion-xl-base-1.0"
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=self.model_path is not None,
        )

        override = str(self.proxy_unet_override_path)
        if override.startswith("hf:"):
            from huggingface_hub import hf_hub_download
            spec = override[3:]  # strip "hf:" prefix
            # Expected format: "<repo_id>/<filename>"; split at LAST slash so
            # repos with org/name (single slash) work cleanly.
            repo_id, filename = spec.rsplit("/", 1)
            unet_ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            unet_ckpt = override

        # DMD2-style UNet ships as a plain state-dict .bin; safetensors also OK.
        if unet_ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(unet_ckpt)
        else:
            state = torch.load(unet_ckpt, map_location="cpu")

        pipeline.unet.load_state_dict(state)
        pipeline.unet = pipeline.unet.to(torch.float16)
        pipeline = pipeline.to(self.device)
        return pipeline

    def _load_ltx_distilled(self) -> LTXPipeline:
        """Load LTX-Video distilled proxy via single-file safetensors.

        The single-file checkpoint only contains the transformer; T5 text encoder,
        tokenizer, VAE, and scheduler must be loaded from the same parent directory
        (which must contain the full diffusers layout from snapshot_download).
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
        """Load Wan2.1-T2V (1.3B) — same loader as teacher (no separate distilled proxy)."""
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
        Load proxy model from a local path.

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

    def get_latent_shape(
        self,
        image_size: int = 1024,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> tuple:
        """
        Get the latent tensor shape for this model family.

        For image models: uses `image_size` (square). Returns 4D (1, C, H, W).
        For LTX (video): uses `height`, `width`, `num_frames`. Returns 5D
        (1, 128, T_latent, H/32, W/32) where T_latent = (num_frames - 1) // 8 + 1.
        """
        latent_size = image_size // 8  # VAE downsampling factor for image models

        if self.model_family == "flux":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sd3":
            return (1, 16, latent_size, latent_size)
        elif self.model_family == "sdxl":
            return (1, 4, latent_size, latent_size)
        elif self.model_family == "ltx":
            assert height is not None and width is not None and num_frames is not None, \
                "LTX latent shape requires height, width, and num_frames"
            h_lat = height // 32
            w_lat = width // 32
            t_lat = (num_frames - 1) // 8 + 1
            return (1, 128, t_lat, h_lat, w_lat)

    def get_default_num_steps(self) -> int:
        """Get the default number of inference steps for this proxy."""
        if self.model_family == "ltx":
            return 8  # LTX distilled default
        if self.model_family == "sdxl":
            return 4
        return 4

    def get_default_guidance_scale(self) -> float:
        """Get the default guidance scale for this proxy."""
        if self.model_family == "flux":
            return 0.0
        elif self.model_family == "sd3":
            return 0.0
        elif self.model_family == "sdxl":
            return 0.0
        elif self.model_family == "ltx":
            # LTX distilled has no CFG / STG
            return 1.0
