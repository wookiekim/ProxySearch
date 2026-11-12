"""Verifier model loading and ensemble scoring."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

# ImageReward runs in a separate venv due to transformers version
# incompatibility.  Scoring is done via subprocess; point
# PROXYSEARCH_IR_PYTHON at that venv's python (see INSTALL.md).
_IR_VENV_PYTHON = os.environ.get("PROXYSEARCH_IR_PYTHON", "/opt/venv_ir/bin/python")
_IR_SCORE_SCRIPT = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "score_image_reward.py")
IMAGE_REWARD_AVAILABLE = Path(_IR_VENV_PYTHON).exists()


def load_clip_model(
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
    device: str = "cuda",
    model_path: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Load an OpenCLIP model for text-image similarity scoring.

    Args:
        model_name: CLIP model architecture name.
        pretrained: Pretrained weights identifier.
        device: Target device.
        model_path: Local path to model weights (optional).

    Returns:
        Tuple of (model, preprocess, tokenizer).

    Raises:
        ImportError: If open_clip is not installed.
    """
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required for CLIP scoring. Install with: pip install open-clip-torch")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained if model_path is None else model_path,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model.eval()
    return model, preprocess, tokenizer


def score_image_reward_subprocess(
    images: List[Image.Image],
    prompt: str,
    device: str = "cuda",
) -> np.ndarray:
    """
    Score images using ImageReward via subprocess (separate venv).

    Args:
        images: List of PIL images.
        prompt: Text prompt for scoring.
        device: Target device.

    Returns:
        Array of ImageReward scores.

    Raises:
        ImportError: If the IR venv is not set up.
        RuntimeError: If the subprocess fails.
    """
    import json
    import subprocess
    import tempfile

    if not IMAGE_REWARD_AVAILABLE:
        raise ImportError(
            f"ImageReward venv python not found at {_IR_VENV_PYTHON}. "
            "Build it with scripts/setup/build_venv_ir.sh and set "
            "PROXYSEARCH_IR_PYTHON to its python."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save images as numbered PNGs
        for i, img in enumerate(images):
            img.convert("RGB").save(Path(tmpdir) / f"{i}.png")

        result = subprocess.run(
            [
                _IR_VENV_PYTHON,
                _IR_SCORE_SCRIPT,
                "--image-dir", tmpdir,
                "--prompt", prompt,
                "--device", device,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ImageReward scoring failed: {result.stderr}")

        # Parse the last line of stdout as JSON (skip any warnings on earlier lines)
        output_lines = result.stdout.strip().split("\n")
        scores = json.loads(output_lines[-1])

    return np.array(scores)


_AESTHETIC_WEIGHTS_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
_AESTHETIC_WEIGHTS_PATH = "/root/.cache/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth"


class AestheticPredictor(torch.nn.Module):
    """LAION aesthetic predictor MLP (768→1024→128→64→16→1 with ReLU)."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_aesthetic_predictor(
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """
    Load LAION aesthetic predictor (v1: MLP on CLIP ViT-L/14 features).

    Outputs scores ~1-10 (typically ~5-7 for decent images).
    Weights auto-downloaded from GitHub on first use.

    Args:
        model_path: Local path to MLP weights. If None, auto-downloads.
        device: Target device.

    Returns:
        Tuple of (aesthetic_model, (clip_model, preprocess)).
    """
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required for aesthetic scoring")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()

    aesthetic_model = AestheticPredictor().to(device)

    weights_path = model_path or _AESTHETIC_WEIGHTS_PATH
    if not Path(weights_path).exists():
        import urllib.request
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_AESTHETIC_WEIGHTS_URL, weights_path)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.eval()

    return aesthetic_model, (clip_model, preprocess)


def score_brightness(images: List[Image.Image]) -> np.ndarray:
    """
    Score images by mean brightness (mean pixel intensity).

    Fast, model-free verifier useful for smoke testing and as a
    simple baseline reward signal.

    Returns:
        Array of scores in [0, 1] range.
    """
    scores = []
    for img in images:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        scores.append(arr.mean() / 255.0)
    return np.array(scores)


def score_compressibility(
    images: List[Image.Image],
    jpeg_quality: int = 95,
) -> np.ndarray:
    """
    Score images by incompressibility (JPEG file size).

    Higher score = larger JPEG = more detail/complexity = less compressible.
    This follows Ma et al. (CVPR 2025) who show that searching for less
    compressible images produces higher-quality outputs.

    Fast, model-free verifier useful for smoke testing.

    Args:
        images: List of PIL images.
        jpeg_quality: JPEG encoding quality (higher = less compression).

    Returns:
        Array of scores (JPEG size in KB).
    """
    import io

    scores = []
    for img in images:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
        size_kb = len(buf.getvalue()) / 1024.0
        scores.append(size_kb)
    return np.array(scores)


def score_contrast(images: List[Image.Image]) -> np.ndarray:
    """Standard deviation of pixel intensities (higher = more contrast)."""
    scores = []
    for img in images:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        scores.append(float(np.std(arr)))
    return np.array(scores)


def score_colorfulness(images: List[Image.Image]) -> np.ndarray:
    """Hasler & Süsstrunk colorfulness metric."""
    scores = []
    for img in images:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        std_root = np.sqrt(std_rg**2 + std_yb**2)
        mean_root = np.sqrt(mean_rg**2 + mean_yb**2)
        scores.append(float(std_root + 0.3 * mean_root))
    return np.array(scores)


def score_saturation(images: List[Image.Image]) -> np.ndarray:
    """Mean saturation in HSV space."""
    scores = []
    for img in images:
        arr = np.array(img.convert("HSV"), dtype=np.float32)
        scores.append(float(np.mean(arr[:, :, 1]) / 255.0))
    return np.array(scores)


def score_sharpness(images: List[Image.Image]) -> np.ndarray:
    """Laplacian variance (higher = sharper)."""
    scores = []
    for img in images:
        arr = np.array(img.convert("L"), dtype=np.float64)
        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.signal import convolve2d
        lap = convolve2d(arr, laplacian, mode="valid")
        scores.append(float(np.var(lap)))
    return np.array(scores)


def score_edge_density(images: List[Image.Image]) -> np.ndarray:
    """Fraction of edge pixels using Sobel gradient threshold."""
    scores = []
    for img in images:
        arr = np.array(img.convert("L"), dtype=np.float64)
        # Sobel kernels
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        from scipy.signal import convolve2d
        gx = convolve2d(arr, sx, mode="valid")
        gy = convolve2d(arr, sy, mode="valid")
        mag = np.sqrt(gx**2 + gy**2)
        threshold = np.mean(mag) + np.std(mag)
        scores.append(float(np.mean(mag > threshold)))
    return np.array(scores)


def score_color_temperature(images: List[Image.Image]) -> np.ndarray:
    """Warm vs cool: log ratio of red to blue channel means. Positive = warm."""
    scores = []
    for img in images:
        arr = np.array(img.convert("RGB"), dtype=np.float32) + 1e-6
        r_mean = np.mean(arr[:, :, 0])
        b_mean = np.mean(arr[:, :, 2])
        scores.append(float(np.log(r_mean / b_mean)))
    return np.array(scores)


class VerifierEnsemble:
    """
    Ensemble of reward verifiers for candidate scoring.

    Combines multiple verifiers (CLIP, ImageReward, Aesthetic, brightness,
    compressibility, and low-freq image statistics) with configurable weights
    to produce an aggregate score.
    """

    SUPPORTED_VERIFIERS = [
        "clip", "image_reward", "aesthetic", "brightness", "compressibility",
        "contrast", "colorfulness", "saturation", "sharpness", "edge_density", "color_temperature",
        "pickscore", "hpsv2",
    ]

    def __init__(
        self,
        verifiers: List[str] = None,
        weights: Optional[List[float]] = None,
        device: str = "cuda",
        clip_model_name: str = "ViT-H-14",
        clip_pretrained: str = "laion2b_s32b_b79k",
        model_paths: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the verifier ensemble.

        Args:
            verifiers: List of verifier names to use. Default: ["clip", "image_reward"].
            weights: Aggregation weights (must match verifiers length).
                    Default: equal weights.
            device: Target device for model loading.
            clip_model_name: CLIP model architecture for CLIP and aesthetic scoring.
            clip_pretrained: Pretrained weights for CLIP.
            model_paths: Dict mapping verifier names to local model paths.

        Raises:
            ValueError: If unsupported verifier is specified or weights mismatch.
        """
        if verifiers is None:
            verifiers = ["clip", "image_reward"]

        for v in verifiers:
            if v not in self.SUPPORTED_VERIFIERS:
                raise ValueError(f"Unsupported verifier: {v}. Supported: {self.SUPPORTED_VERIFIERS}")

        if weights is None:
            weights = [1.0 / len(verifiers)] * len(verifiers)

        if len(weights) != len(verifiers):
            raise ValueError(f"Weights length ({len(weights)}) must match verifiers length ({len(verifiers)})")

        self.verifier_names = verifiers
        self.weights = np.array(weights)
        self.device = device
        self.model_paths = model_paths or {}

        # Lazy loading - models loaded on first use
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._image_reward_model = None
        self._aesthetic_model = None
        self._aesthetic_clip = None
        # Learned preference models (PickScore, HPSv2). Loaded on first use.
        self._pickscore_model = None
        self._pickscore_processor = None
        self._hpsv2_initialized = False

        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained

    def _ensure_clip_loaded(self) -> None:
        """Ensure CLIP model is loaded."""
        if self._clip_model is None:
            self._clip_model, self._clip_preprocess, self._clip_tokenizer = load_clip_model(
                model_name=self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=self.device,
                model_path=self.model_paths.get("clip"),
            )

    def _ensure_image_reward_loaded(self) -> None:
        """Check ImageReward availability (loaded via subprocess, no model to hold)."""
        if not IMAGE_REWARD_AVAILABLE:
            raise ImportError(f"ImageReward venv python not found at {_IR_VENV_PYTHON}")

    def _ensure_aesthetic_loaded(self) -> None:
        """Ensure aesthetic predictor is loaded."""
        if self._aesthetic_model is None:
            self._aesthetic_model, self._aesthetic_clip = load_aesthetic_predictor(
                model_path=self.model_paths.get("aesthetic"),
                device=self.device,
            )

    def _score_clip(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> np.ndarray:
        """Compute CLIP scores for images and prompts."""
        self._ensure_clip_loaded()

        with torch.no_grad():
            # Preprocess images
            image_tensors = torch.stack([
                self._clip_preprocess(img) for img in images
            ]).to(self.device)

            # Tokenize prompts
            text_tokens = self._clip_tokenizer(prompts).to(self.device)

            # Get embeddings
            image_features = self._clip_model.encode_image(image_tensors)
            text_features = self._clip_model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            scores = (image_features * text_features).sum(dim=-1)

        return scores.cpu().numpy()

    def _score_image_reward(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> np.ndarray:
        """Compute ImageReward scores via subprocess."""
        self._ensure_image_reward_loaded()

        # ImageReward scores all images against a single prompt,
        # so group by prompt for efficiency
        unique_prompts = list(dict.fromkeys(prompts))
        if len(unique_prompts) == 1:
            return score_image_reward_subprocess(images, unique_prompts[0], device=self.device)

        # Multiple prompts: score per-prompt groups
        scores = np.zeros(len(images))
        prompt_to_indices = {}
        for i, p in enumerate(prompts):
            prompt_to_indices.setdefault(p, []).append(i)

        for prompt, indices in prompt_to_indices.items():
            group_images = [images[i] for i in indices]
            group_scores = score_image_reward_subprocess(group_images, prompt, device=self.device)
            for idx, score in zip(indices, group_scores):
                scores[idx] = score

        return scores

    def _score_aesthetic(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> np.ndarray:
        """Compute aesthetic scores for images."""
        self._ensure_aesthetic_loaded()

        clip_model, preprocess = self._aesthetic_clip

        with torch.no_grad():
            # Get CLIP image features
            image_tensors = torch.stack([
                preprocess(img) for img in images
            ]).to(self.device)

            image_features = clip_model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Predict aesthetic score (MLP expects float32)
            scores = self._aesthetic_model(image_features.float()).squeeze(-1)

        return scores.cpu().numpy()

    # ------------------------------------------------------------------
    # Learned preference models: PickScore, HPSv2.
    # Both follow CLIP-H based dual-encoder + cosine-similarity scoring.
    # PickScore: yuvalkirstain/PickScore_v1, loadable as transformers AutoModel.
    # HPSv2: via the hpsv2 pip package (pre-installed); calls hpsv2.score().
    # ------------------------------------------------------------------
    def _ensure_pickscore_loaded(self) -> None:
        """Ensure PickScore model + processor are loaded."""
        if self._pickscore_model is None:
            from transformers import AutoProcessor, AutoModel
            processor_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            model_id = "yuvalkirstain/PickScore_v1"
            self._pickscore_processor = AutoProcessor.from_pretrained(processor_id)
            self._pickscore_model = AutoModel.from_pretrained(model_id).eval().to(self.device)

    def _score_pickscore(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> np.ndarray:
        """PickScore_v1 — CLIP-H dual encoder; logits_per_image is the
        logit-scaled cosine similarity (scalar reward used by PickScore)."""
        self._ensure_pickscore_loaded()

        scores = np.zeros(len(images), dtype=np.float32)
        prompt_to_indices: Dict[str, List[int]] = {}
        for i, p in enumerate(prompts):
            prompt_to_indices.setdefault(p, []).append(i)

        with torch.no_grad():
            for prompt, idxs in prompt_to_indices.items():
                group_images = [images[i] for i in idxs]
                image_inputs = self._pickscore_processor(
                    images=group_images, return_tensors="pt"
                ).to(self.device)
                text_inputs = self._pickscore_processor(
                    text=[prompt], padding=True, truncation=True, max_length=77,
                    return_tensors="pt"
                ).to(self.device)
                # CLIPModel forward returns CLIPOutput with logits_per_image
                # (shape: (n_images, n_texts), already logit-scaled cosine).
                combined = {**image_inputs, **text_inputs}
                out = self._pickscore_model(**combined)
                # logits_per_image: (n_images, 1) for our single-prompt case.
                group_scores = out.logits_per_image.squeeze(-1).cpu().numpy().astype(np.float32)
                for local_idx, global_idx in enumerate(idxs):
                    scores[global_idx] = float(group_scores[local_idx])
        return scores

    def _ensure_hpsv2_loaded(self) -> None:
        """Ensure hpsv2 package is importable. Initialization is lazy in hpsv2.score()."""
        if not self._hpsv2_initialized:
            import hpsv2  # noqa: F401  (validates import; weights download on first call)
            self._hpsv2_initialized = True

    def _score_hpsv2(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> np.ndarray:
        """HPSv2 — open_clip H-14 finetuned on human preference pairs.
        Uses the hpsv2 pip package; returns raw HPSv2 score per (image, prompt).
        """
        self._ensure_hpsv2_loaded()
        import hpsv2

        # hpsv2.score() accepts a list of PIL images and a single prompt; returns
        # a list of score floats. Group by prompt for efficiency.
        scores = np.zeros(len(images), dtype=np.float32)
        prompt_to_indices: Dict[str, List[int]] = {}
        for i, p in enumerate(prompts):
            prompt_to_indices.setdefault(p, []).append(i)

        for prompt, idxs in prompt_to_indices.items():
            group_images = [images[i] for i in idxs]
            # hpsv2.score returns a list of floats with one entry per image.
            # API: hpsv2.score(image, prompt, hps_version="v2.1") -> np.ndarray | list
            # Pass a single image at a time to be robust to API variations.
            group_scores = []
            for img in group_images:
                s = hpsv2.score(img, prompt, hps_version="v2.1")
                if isinstance(s, (list, tuple, np.ndarray)):
                    s = float(s[0])
                else:
                    s = float(s)
                group_scores.append(s)
            for local_idx, global_idx in enumerate(idxs):
                scores[global_idx] = group_scores[local_idx]
        return scores

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompts: Union[str, List[str]],
    ) -> Union[float, np.ndarray]:
        """
        Score image(s) against prompt(s).

        Args:
            images: Single PIL image or list of images.
            prompts: Single prompt string or list of prompts.

        Returns:
            Aggregated verifier score(s). Returns float for single image,
            numpy array for multiple images.
        """
        # Handle single inputs
        single_input = isinstance(images, Image.Image)
        if single_input:
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        # Ensure we have matching lengths
        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)

        # Compute scores from each verifier
        all_scores = []
        for verifier_name in self.verifier_names:
            if verifier_name == "clip":
                scores = self._score_clip(images, prompts)
            elif verifier_name == "image_reward":
                scores = self._score_image_reward(images, prompts)
            elif verifier_name == "aesthetic":
                scores = self._score_aesthetic(images, prompts)
            elif verifier_name == "brightness":
                scores = score_brightness(images)
            elif verifier_name == "compressibility":
                scores = score_compressibility(images)
            elif verifier_name == "contrast":
                scores = score_contrast(images)
            elif verifier_name == "colorfulness":
                scores = score_colorfulness(images)
            elif verifier_name == "saturation":
                scores = score_saturation(images)
            elif verifier_name == "sharpness":
                scores = score_sharpness(images)
            elif verifier_name == "edge_density":
                scores = score_edge_density(images)
            elif verifier_name == "color_temperature":
                scores = score_color_temperature(images)
            elif verifier_name == "pickscore":
                scores = self._score_pickscore(images, prompts)
            elif verifier_name == "hpsv2":
                scores = self._score_hpsv2(images, prompts)
            all_scores.append(scores)

        # Stack and compute weighted sum
        all_scores = np.stack(all_scores, axis=0)  # (num_verifiers, num_images)
        aggregated = np.sum(all_scores * self.weights[:, None], axis=0)

        if single_input:
            return float(aggregated[0])
        return aggregated

    def score_individual(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Return per-verifier scores for analysis.

        Args:
            images: List of PIL images.
            prompts: List of prompts (one per image or single prompt for all).

        Returns:
            Dictionary mapping verifier names to score arrays.
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)

        results = {}
        for verifier_name in self.verifier_names:
            if verifier_name == "clip":
                results["clip"] = self._score_clip(images, prompts)
            elif verifier_name == "image_reward":
                results["image_reward"] = self._score_image_reward(images, prompts)
            elif verifier_name == "aesthetic":
                results["aesthetic"] = self._score_aesthetic(images, prompts)
            elif verifier_name == "brightness":
                results["brightness"] = score_brightness(images)
            elif verifier_name == "compressibility":
                results["compressibility"] = score_compressibility(images)
            elif verifier_name == "contrast":
                results["contrast"] = score_contrast(images)
            elif verifier_name == "colorfulness":
                results["colorfulness"] = score_colorfulness(images)
            elif verifier_name == "saturation":
                results["saturation"] = score_saturation(images)
            elif verifier_name == "sharpness":
                results["sharpness"] = score_sharpness(images)
            elif verifier_name == "edge_density":
                results["edge_density"] = score_edge_density(images)
            elif verifier_name == "color_temperature":
                results["color_temperature"] = score_color_temperature(images)
            elif verifier_name == "pickscore":
                results["pickscore"] = self._score_pickscore(images, prompts)
            elif verifier_name == "hpsv2":
                results["hpsv2"] = self._score_hpsv2(images, prompts)

        return results

    def unload(self) -> None:
        """Unload all models from memory."""
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._image_reward_model = None
        self._aesthetic_model = None
        self._aesthetic_clip = None
        self._pickscore_model = None
        self._pickscore_processor = None
        self._hpsv2_initialized = False
        torch.cuda.empty_cache()
