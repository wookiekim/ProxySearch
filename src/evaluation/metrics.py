"""Core evaluation metrics for image generation."""

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

try:
    import ImageReward as image_reward_lib

    IMAGE_REWARD_AVAILABLE = True
except ImportError:
    IMAGE_REWARD_AVAILABLE = False


def clip_score(
    images: Union[Image.Image, List[Image.Image]],
    prompts: Union[str, List[str]],
    model: Optional[Any] = None,
    preprocess: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
    device: str = "cuda",
) -> Union[float, np.ndarray]:
    """
    Compute CLIP score (cosine similarity) between images and prompts.

    Args:
        images: Single PIL image or list of images.
        prompts: Single prompt or list of prompts.
        model: Pre-loaded CLIP model (optional).
        preprocess: Pre-loaded preprocessing function (optional).
        tokenizer: Pre-loaded tokenizer (optional).
        model_name: CLIP model architecture if loading fresh.
        pretrained: Pretrained weights identifier.
        device: Target device.

    Returns:
        CLIP score(s) as float or numpy array.
    """
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required. Install with: pip install open-clip-torch")

    # Load model if not provided
    if model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()

    # Handle single inputs
    single_input = isinstance(images, Image.Image)
    if single_input:
        images = [images]
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)

    with torch.no_grad():
        # Preprocess images
        image_tensors = torch.stack([preprocess(img) for img in images]).to(device)

        # Tokenize prompts
        text_tokens = tokenizer(prompts).to(device)

        # Get embeddings
        image_features = model.encode_image(image_tensors)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        scores = (image_features * text_features).sum(dim=-1)

    scores = scores.cpu().numpy()

    if single_input:
        return float(scores[0])
    return scores


def image_reward_score(
    images: Union[Image.Image, List[Image.Image]],
    prompts: Union[str, List[str]],
    model: Optional[Any] = None,
    device: str = "cuda",
) -> Union[float, np.ndarray]:
    """
    Compute ImageReward score for images.

    Args:
        images: Single PIL image or list of images.
        prompts: Single prompt or list of prompts.
        model: Pre-loaded ImageReward model (optional).
        device: Target device.

    Returns:
        ImageReward score(s) as float or numpy array.
    """
    if not IMAGE_REWARD_AVAILABLE:
        raise ImportError("ImageReward is required. Install with: pip install image-reward")

    # Load model if not provided
    if model is None:
        model = image_reward_lib.load("ImageReward-v1.0", device=device)

    # Handle single inputs
    single_input = isinstance(images, Image.Image)
    if single_input:
        images = [images]
    if isinstance(prompts, str):
        prompts = [prompts] * len(images)

    scores = []
    for img, prompt in zip(images, prompts):
        score = model.score(prompt, img)
        scores.append(score)

    scores = np.array(scores)

    if single_input:
        return float(scores[0])
    return scores


def compute_fid(
    generated_images: Union[List[Image.Image], str, Path],
    reference_images: Union[List[Image.Image], str, Path],
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 4,
) -> float:
    """
    Compute Frechet Inception Distance between generated and reference images.

    Args:
        generated_images: List of PIL images or path to directory.
        reference_images: List of PIL images or path to directory.
        batch_size: Batch size for feature extraction.
        device: Target device.
        num_workers: Number of workers for data loading.

    Returns:
        FID score (lower is better).
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        raise ImportError("pytorch_fid is required. Install with: pip install pytorch-fid")

    import tempfile
    import os

    # If inputs are lists of images, save them to temporary directories
    temp_dirs = []

    def prepare_images(images):
        if isinstance(images, (str, Path)):
            return str(images)

        # Save images to temp directory
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)

        for i, img in enumerate(images):
            img.save(os.path.join(temp_dir, f"{i:06d}.png"))

        return temp_dir

    generated_path = prepare_images(generated_images)
    reference_path = prepare_images(reference_images)

    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [generated_path, reference_path],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=num_workers,
        )
    finally:
        # Cleanup temp directories
        import shutil
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return float(fid_value)


def compute_inception_score(
    images: Union[List[Image.Image], str, Path],
    batch_size: int = 64,
    splits: int = 10,
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    Compute Inception Score for generated images.

    Args:
        images: List of PIL images or path to directory.
        batch_size: Batch size for feature extraction.
        splits: Number of splits for computing mean/std.
        device: Target device.

    Returns:
        Tuple of (mean IS, std IS).
    """
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required for Inception Score")

    import tempfile
    import os

    # Load Inception model (use weights parameter for newer torchvision)
    try:
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    except TypeError:
        # Fallback for older torchvision versions
        model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Handle directory input
    if isinstance(images, (str, Path)):
        image_dir = Path(images)
        images = [Image.open(p).convert("RGB") for p in sorted(image_dir.glob("*.png"))]

    # Get predictions
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch]).to(device)
            pred = torch.nn.functional.softmax(model(batch_tensor), dim=1)
            preds.append(pred.cpu())

    preds = torch.cat(preds, dim=0).numpy()

    # Compute IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits) : (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def compute_lpips(
    images1: List[Image.Image],
    images2: List[Image.Image],
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute LPIPS perceptual distance between pairs of images.

    Args:
        images1: First list of images.
        images2: Second list of images (same length as images1).
        device: Target device.

    Returns:
        Array of LPIPS distances.
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips is required. Install with: pip install lpips")

    from torchvision import transforms

    model = lpips.LPIPS(net='alex').to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    distances = []
    with torch.no_grad():
        for img1, img2 in zip(images1, images2):
            t1 = preprocess(img1).unsqueeze(0).to(device)
            t2 = preprocess(img2).unsqueeze(0).to(device)
            d = model(t1, t2)
            distances.append(d.item())

    return np.array(distances)


class MetricsComputer:
    """
    Unified interface for computing all metrics.

    Handles model caching and batch computation for efficiency.
    """

    def __init__(
        self,
        device: str = "cuda",
        clip_model_name: str = "ViT-H-14",
        clip_pretrained: str = "laion2b_s32b_b79k",
    ):
        """
        Initialize the metrics computer.

        Args:
            device: Target device.
            clip_model_name: CLIP model architecture.
            clip_pretrained: CLIP pretrained weights.
        """
        self.device = device
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained

        # Lazy loading
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._image_reward_model = None

    def _ensure_clip_loaded(self) -> None:
        """Ensure CLIP model is loaded."""
        if self._clip_model is None and OPEN_CLIP_AVAILABLE:
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=self.device,
            )
            self._clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            self._clip_model.eval()

    def _ensure_image_reward_loaded(self) -> None:
        """Ensure ImageReward model is loaded."""
        if self._image_reward_model is None and IMAGE_REWARD_AVAILABLE:
            self._image_reward_model = image_reward_lib.load(
                "ImageReward-v1.0", device=self.device
            )

    def compute_all(
        self,
        images: List[Image.Image],
        prompts: List[str],
        reference_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all available metrics.

        Args:
            images: List of generated images.
            prompts: List of prompts (one per image or one for all).
            reference_images: Optional reference images for FID.

        Returns:
            Dictionary with all metrics.
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        results = {}

        # CLIP score
        try:
            self._ensure_clip_loaded()
            results["clip_scores"] = clip_score(
                images,
                prompts,
                model=self._clip_model,
                preprocess=self._clip_preprocess,
                tokenizer=self._clip_tokenizer,
                device=self.device,
            )
            results["clip_mean"] = float(np.mean(results["clip_scores"]))
            results["clip_std"] = float(np.std(results["clip_scores"]))
        except (ImportError, Exception) as e:
            results["clip_error"] = str(e)

        # ImageReward
        try:
            self._ensure_image_reward_loaded()
            results["image_reward_scores"] = image_reward_score(
                images,
                prompts,
                model=self._image_reward_model,
                device=self.device,
            )
            results["image_reward_mean"] = float(np.mean(results["image_reward_scores"]))
            results["image_reward_std"] = float(np.std(results["image_reward_scores"]))
        except (ImportError, Exception) as e:
            results["image_reward_error"] = str(e)

        # FID
        if reference_images is not None:
            try:
                results["fid"] = compute_fid(
                    images, reference_images, device=self.device
                )
            except (ImportError, Exception) as e:
                results["fid_error"] = str(e)

        return results

    def unload(self) -> None:
        """Unload all models."""
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._image_reward_model = None
        torch.cuda.empty_cache()
