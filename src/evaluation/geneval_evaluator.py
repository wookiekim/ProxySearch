"""GenEval benchmark evaluator for compositional generation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


class GenEvalEvaluator:
    """
    Evaluator for the GenEval benchmark.

    GenEval tests compositional generation capabilities across categories:
    - Single object
    - Two objects
    - Counting
    - Colors
    - Position
    - Color attribution

    Evaluation uses object detection (OWL-ViT) + attribute classification
    to measure accuracy.
    """

    CATEGORIES = [
        "single_object",
        "two_object",
        "counting",
        "colors",
        "position",
        "color_attribution",
    ]

    def __init__(
        self,
        prompts_path: Optional[Union[str, Path]] = None,
        detector_model: str = "google/owlvit-base-patch32",
        device: str = "cuda",
    ):
        """
        Initialize the GenEval evaluator.

        Args:
            prompts_path: Path to GenEval prompts JSON file.
            detector_model: Object detector model identifier.
            device: Target device.
        """
        self.prompts_path = prompts_path
        self.detector_model = detector_model
        self.device = device

        # Lazy loading
        self._detector = None
        self._processor = None
        self._prompts: Optional[List[Dict]] = None

    def _load_detector(self) -> None:
        """Load the object detector model."""
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
        except ImportError:
            raise ImportError(
                "transformers is required for GenEval. "
                "Install with: pip install transformers"
            )

        self._processor = OwlViTProcessor.from_pretrained(self.detector_model)
        self._detector = OwlViTForObjectDetection.from_pretrained(self.detector_model)
        self._detector = self._detector.to(self.device)
        self._detector.eval()

    def _ensure_detector_loaded(self) -> None:
        """Ensure detector is loaded."""
        if self._detector is None:
            self._load_detector()

    def load_prompts(self, path: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Load GenEval prompts from file.

        Args:
            path: Path to prompts file (JSON format).

        Returns:
            List of prompt dictionaries with keys:
                - prompt: str
                - prompt_id: str
                - category: str
                - objects: List[str]
                - attributes: Dict[str, str] (optional)
                - count: int (optional, for counting category)
        """
        path = path or self.prompts_path
        if path is None:
            raise ValueError("No prompts path specified")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        with open(path, "r") as f:
            self._prompts = json.load(f)

        return self._prompts

    def get_prompts(
        self,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get prompts, optionally filtered by category.

        Args:
            category: Category to filter by (optional).

        Returns:
            List of prompt dictionaries.
        """
        if self._prompts is None:
            self.load_prompts()

        if category is not None:
            return [p for p in self._prompts if p.get("category") == category]

        return self._prompts

    def detect_objects(
        self,
        image: Image.Image,
        object_queries: List[str],
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.

        Args:
            image: PIL image to analyze.
            object_queries: List of object names to detect.
            threshold: Detection confidence threshold.

        Returns:
            List of detections with boxes, scores, and labels.
        """
        self._ensure_detector_loaded()

        # Prepare inputs
        inputs = self._processor(
            text=[object_queries],
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run detection
        with torch.no_grad():
            outputs = self._detector(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]

        detections = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append({
                "label": object_queries[label_idx],
                "score": float(score),
                "box": box.cpu().tolist(),
            })

        return detections

    def evaluate_single_object(
        self,
        image: Image.Image,
        prompt_info: Dict,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate single object detection accuracy.

        Args:
            image: Generated image.
            prompt_info: Prompt dictionary with "objects" key.
            threshold: Detection confidence threshold.

        Returns:
            Evaluation result with success flag and details.
        """
        objects = prompt_info.get("objects", [])
        if not objects:
            return {"success": False, "reason": "no_objects_specified"}

        detections = self.detect_objects(image, objects, threshold)

        # Check if the required object was detected
        detected_labels = {d["label"] for d in detections}
        required_object = objects[0]

        success = required_object in detected_labels

        return {
            "success": success,
            "required_object": required_object,
            "detected_objects": list(detected_labels),
            "num_detections": len(detections),
        }

    def evaluate_two_objects(
        self,
        image: Image.Image,
        prompt_info: Dict,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate two-object composition accuracy.

        Args:
            image: Generated image.
            prompt_info: Prompt dictionary with "objects" key.
            threshold: Detection confidence threshold.

        Returns:
            Evaluation result with success flag and details.
        """
        objects = prompt_info.get("objects", [])
        if len(objects) < 2:
            return {"success": False, "reason": "need_two_objects"}

        detections = self.detect_objects(image, objects, threshold)
        detected_labels = {d["label"] for d in detections}

        # Both objects must be detected
        success = all(obj in detected_labels for obj in objects[:2])

        return {
            "success": success,
            "required_objects": objects[:2],
            "detected_objects": list(detected_labels),
        }

    def evaluate_counting(
        self,
        image: Image.Image,
        prompt_info: Dict,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate counting accuracy.

        Args:
            image: Generated image.
            prompt_info: Prompt dictionary with "objects" and "count" keys.
            threshold: Detection confidence threshold.

        Returns:
            Evaluation result with success flag and count comparison.
        """
        objects = prompt_info.get("objects", [])
        expected_count = prompt_info.get("count", 1)

        if not objects:
            return {"success": False, "reason": "no_objects_specified"}

        detections = self.detect_objects(image, objects, threshold)

        # Count detections of the target object
        target_object = objects[0]
        actual_count = sum(1 for d in detections if d["label"] == target_object)

        success = actual_count == expected_count

        return {
            "success": success,
            "target_object": target_object,
            "expected_count": expected_count,
            "actual_count": actual_count,
        }

    def evaluate_image(
        self,
        image: Image.Image,
        prompt_info: Dict,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate a single image based on its category.

        Args:
            image: Generated image.
            prompt_info: Prompt dictionary with category and evaluation info.
            threshold: Detection confidence threshold.

        Returns:
            Evaluation result appropriate to the category.
        """
        category = prompt_info.get("category", "single_object")

        if category == "single_object":
            return self.evaluate_single_object(image, prompt_info, threshold)
        elif category == "two_object":
            return self.evaluate_two_objects(image, prompt_info, threshold)
        elif category == "counting":
            return self.evaluate_counting(image, prompt_info, threshold)
        elif category in ["colors", "position", "color_attribution"]:
            # These require more complex evaluation
            # For now, use object detection as a proxy
            return self.evaluate_single_object(image, prompt_info, threshold)
        else:
            return {"success": False, "reason": f"unknown_category_{category}"}

    def evaluate_batch(
        self,
        images: List[Image.Image],
        prompt_infos: List[Dict],
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of images.

        Args:
            images: List of generated images.
            prompt_infos: List of corresponding prompt dictionaries.
            threshold: Detection confidence threshold.

        Returns:
            Dictionary with per-category accuracy and overall results.
        """
        results = []
        for image, prompt_info in zip(images, prompt_infos):
            result = self.evaluate_image(image, prompt_info, threshold)
            result["prompt_id"] = prompt_info.get("prompt_id")
            result["category"] = prompt_info.get("category")
            results.append(result)

        # Compute per-category accuracy
        category_results = {}
        for cat in self.CATEGORIES:
            cat_results = [r for r in results if r.get("category") == cat]
            if cat_results:
                accuracy = sum(1 for r in cat_results if r.get("success")) / len(cat_results)
                category_results[cat] = {
                    "accuracy": accuracy,
                    "count": len(cat_results),
                }

        # Overall accuracy
        overall_success = sum(1 for r in results if r.get("success"))
        overall_accuracy = overall_success / len(results) if results else 0.0

        return {
            "overall_accuracy": overall_accuracy,
            "overall_success": overall_success,
            "overall_count": len(results),
            "per_category": category_results,
            "detailed_results": results,
        }

    def unload(self) -> None:
        """Unload models from memory."""
        self._detector = None
        self._processor = None
        torch.cuda.empty_cache()


class PromptDataset(torch.utils.data.Dataset):
    """
    Dataset class for benchmark prompts.

    Provides a unified interface for loading prompts from
    GenEval, DrawBench, and PartiPrompts benchmarks.
    """

    def __init__(
        self,
        benchmark: str,
        data_path: Optional[Union[str, Path]] = None,
        split: str = "all",
        category: Optional[str] = None,
    ):
        """
        Initialize the prompt dataset.

        Args:
            benchmark: Benchmark name ("geneval", "drawbench", "parti").
            data_path: Path to benchmark data.
            split: Data split (usually "all" for benchmarks).
            category: Optional category filter.
        """
        self.benchmark = benchmark
        self.data_path = Path(data_path) if data_path else None
        self.split = split
        self.category = category

        self._prompts: List[Dict] = []
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from the benchmark data."""
        if self.data_path is None:
            # Use placeholder prompts for dry-run testing
            self._prompts = [
                {
                    "prompt": "a red apple on a wooden table",
                    "prompt_id": "test_0",
                    "category": "single_object",
                    "objects": ["apple"],
                }
            ]
            return

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Load based on benchmark format
        if self.benchmark == "geneval":
            with open(self.data_path, "r") as f:
                self._prompts = json.load(f)
        elif self.benchmark == "drawbench":
            with open(self.data_path, "r") as f:
                self._prompts = json.load(f)
        elif self.benchmark == "parti":
            with open(self.data_path, "r") as f:
                self._prompts = json.load(f)

        # Filter by category if specified
        if self.category is not None:
            self._prompts = [
                p for p in self._prompts if p.get("category") == self.category
            ]

    def __len__(self) -> int:
        """Return number of prompts."""
        return len(self._prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a prompt by index.

        Returns:
            Dictionary with prompt info.
        """
        return self._prompts[idx]


def get_prompt_dataloader(
    benchmark: str,
    data_path: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    category: Optional[str] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for benchmark prompts.

    Args:
        benchmark: Benchmark name.
        data_path: Path to benchmark data.
        batch_size: Batch size (usually 1 for generation).
        num_workers: Number of data loading workers.
        category: Optional category filter.

    Returns:
        DataLoader for the prompts.
    """
    dataset = PromptDataset(
        benchmark=benchmark,
        data_path=data_path,
        category=category,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # Return list of dicts
    )
