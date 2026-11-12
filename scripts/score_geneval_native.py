#!/usr/bin/env python3
"""Native GenEval scorer — HF transformers Mask2Former + open_clip. No mmcv/mmdet.

Drop-in replacement for `scripts/score_geneval.py` (which requires the GenEval venv
with mmcv 1.7.2 + mmdet 2.28.2). Runs in the main environment.

Produces the same evaluate(image, detected, metadata) → (correct, reason) logic as
GenEval's official pipeline, using `facebook/mask2former-swin-small-coco-instance`
in place of the mmdet Mask2Former checkpoint. Label mapping included for the 6
class names that differ between HF (PASCAL-VOC-style) and COCO canonical.

Public API:

    from score_geneval_native import NativeGenEvalScorer
    scorer = NativeGenEvalScorer(device="cuda")
    result = scorer.score(pil_image, metadata_dict)
    # result = {"success": bool, "correct": bool, "reason": str,
    #          "fraction": float, "detected_summary": dict}

    scores = scorer.score_batch(pil_images, metadata_dict)
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


# --- HF Mask2Former COCO instance → COCO canonical label mapping ---

# GenEval's object_names.txt uses COCO canonical names; HF checkpoint uses
# PASCAL-VOC-style for 6 classes. Map HF label → canonical.
HF_TO_COCO = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}

COCO_CLASSNAMES: List[str] = []  # populated at load from object_names.txt


# --- Official GenEval evaluate helpers — ported verbatim ---

COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "black", "white"]
POSITION_THRESHOLD = 0.1


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """A relative to B. Returns set of relations."""
    boxes = np.array([obj_a[0][:4], obj_b[0][:4]]).reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    norm = np.linalg.norm(offset) or 1e-8
    dx, dy = revised / norm
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform, bgcolor="#999", crop=True):
        self._image = image.convert("RGB")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform
        self._crop = crop

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, idx):
        box, mask = self._objects[idx]
        if mask is not None:
            image = Image.composite(self._image, self._blank, Image.fromarray(mask.astype(np.uint8) * 255))
        else:
            image = self._image
        if self._crop:
            image = image.crop(tuple(box[:4].astype(int)))
        return self._transform(image), 0


class NativeGenEvalScorer:
    """In-process GenEval pipeline — Mask2Former + CLIP — main venv native."""

    def __init__(
        self,
        device: str = "cuda",
        detector_model: str = "facebook/mask2former-swin-small-coco-instance",
        clip_arch: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        threshold: float = 0.3,
        counting_threshold: float = 0.9,
        max_objects: int = 16,
        nms_threshold: float = 1.0,
        object_names_file: Optional[Path] = None,
    ):
        self.device = device
        self.threshold = threshold
        self.counting_threshold = counting_threshold
        self.max_objects = max_objects
        self.nms_threshold = nms_threshold

        # Load canonical class names
        global COCO_CLASSNAMES
        onf = Path(object_names_file) if object_names_file else Path(
            __file__).parent.parent / "benchmarks" / "geneval" / "evaluation" / "object_names.txt"
        with open(onf) as f:
            COCO_CLASSNAMES = [line.strip() for line in f if line.strip()]
        assert len(COCO_CLASSNAMES) == 80, f"expected 80 COCO classes, got {len(COCO_CLASSNAMES)}"

        # Mask2Former
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        print(f"[geneval_native] loading {detector_model}", file=sys.stderr)
        self.detector_processor = Mask2FormerImageProcessor.from_pretrained(detector_model)
        self.detector = Mask2FormerForUniversalSegmentation.from_pretrained(
            detector_model).to(device).eval()
        # HF id2label → canonical COCO name mapping
        self.id_to_coco: Dict[int, str] = {}
        for i, l in self.detector.config.id2label.items():
            self.id_to_coco[int(i)] = HF_TO_COCO.get(l, l)

        # CLIP
        import open_clip
        print(f"[geneval_native] loading CLIP {clip_arch} {clip_pretrained}", file=sys.stderr)
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained=clip_pretrained, device=device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_arch)
        self.clip_model.eval()
        self._color_classifiers: Dict[str, Any] = {}

    # --- detection ---

    def _detect(self, image: Image.Image, counting: bool) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Run Mask2Former, return {canonical_classname: [(bbox[xyxy,score], mask_2d), ...]}.

        bbox is np.array([x1, y1, x2, y2, score]).
        mask is boolean np.array of same HxW as image, or None if no mask.
        """
        conf_threshold = self.counting_threshold if counting else self.threshold
        W, H = image.size

        # Processor inputs
        inputs = self.detector_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.detector(**inputs)
        # Instance segmentation post-processing (returns per-instance masks + class + score)
        results = self.detector_processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(H, W)],
            threshold=0.0,  # we filter with conf_threshold below; keep all here to rank
            return_binary_maps=True,
        )[0]

        # results["segmentation"]: (num_instances, H, W) boolean
        # results["segments_info"]: list of {"id", "label_id", "score", ...}
        segs = results.get("segmentation")
        infos = results.get("segments_info", [])

        if segs is None or len(infos) == 0:
            return {}

        # Move to cpu numpy
        if isinstance(segs, torch.Tensor):
            segs_np = segs.detach().cpu().numpy().astype(bool)
        else:
            segs_np = np.array(segs).astype(bool)

        # Organize by class, sort by score, apply NMS
        by_class: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        # The processor returns segments in order; map each to its index into segs_np
        for info in infos:
            score = float(info.get("score", 0.0))
            if score < conf_threshold:
                continue
            lid = int(info["label_id"])
            canon = self.id_to_coco.get(lid)
            if canon is None:
                continue
            mask_idx = int(info["id"]) - 1 if info.get("id", 0) > 0 else None
            # Processor uses segments_info order in some versions; fallback to enumerate
            # Safer: locate the mask by checking which channel is non-empty for this segment
            # HF returns segments_info with "id" that matches value in segmentation map OR channel idx
            # Use segs_np[i] for i in range(len(infos)) assumption — matches HF docs when return_binary_maps=True
            # Fall back: just use position in list
            # Since return_binary_maps=True returns (num_segments, H, W) aligned with segments_info,
            # we use enumerate below. Adjust logic:
            pass
        # Redo aligned loop
        by_class = {}
        for ch_i, info in enumerate(infos):
            score = float(info.get("score", 0.0))
            if score < conf_threshold:
                continue
            lid = int(info["label_id"])
            canon = self.id_to_coco.get(lid)
            if canon is None:
                continue
            if ch_i >= segs_np.shape[0]:
                continue
            mask = segs_np[ch_i]
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            box = np.array([x1, y1, x2, y2, score], dtype=np.float32)
            by_class.setdefault(canon, []).append((box, mask))

        # Sort per-class by score desc; apply NMS; limit max_objects
        out: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for canon, objs in by_class.items():
            objs.sort(key=lambda t: -t[0][4])
            kept: List[Tuple[np.ndarray, np.ndarray]] = []
            for obj in objs:
                if any(
                    compute_iou(obj[0], k[0]) >= self.nms_threshold
                    for k in kept
                ) and self.nms_threshold < 1.0:
                    continue
                kept.append(obj)
                if len(kept) >= self.max_objects:
                    break
            if kept:
                out[canon] = kept
        return out

    # --- color classification (CLIP zero-shot) ---

    def _color_classify(self, image: Image.Image, bboxes: List[Tuple[np.ndarray, np.ndarray]],
                         classname: str) -> List[str]:
        if classname not in self._color_classifiers:
            prompts = [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object",
            ]
            # Build zero-shot classifier: (num_classes, num_prompts) text embeds averaged
            with torch.no_grad():
                class_embeds = []
                for color in COLORS:
                    texts = [p.format(c=color) for p in prompts]
                    tokens = self.clip_tokenizer(texts).to(self.device)
                    feats = self.clip_model.encode_text(tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    class_embeds.append(feats.mean(dim=0))
                classifier = torch.stack(class_embeds)  # (num_colors, D)
                classifier = classifier / classifier.norm(dim=-1, keepdim=True)
            self._color_classifiers[classname] = classifier

        clf = self._color_classifiers[classname]
        ds = ImageCrops(image, bboxes, self.clip_transform)
        if len(ds) == 0:
            return []
        dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0)
        preds: List[str] = []
        with torch.no_grad():
            for imgs, _ in dl:
                imgs = imgs.to(self.device)
                feats = self.clip_model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = feats @ clf.T
                idx = logits.argmax(dim=1).cpu().numpy()
                preds.extend([COLORS[int(i)] for i in idx])
        return preds

    # --- official evaluate() port ---

    def _evaluate(self, image: Image.Image, detected: Dict[str, List], metadata: Dict) -> Tuple[bool, str, int, int]:
        """Return (is_correct, reason_str, num_reqs, num_failed)."""
        correct = True
        reasons: List[str] = []
        matched_groups: List[Optional[List]] = []
        include = metadata.get("include", [])
        exclude = metadata.get("exclude", [])

        for req in include:
            classname = req["class"]
            matched = True
            found = detected.get(classname, [])[: req["count"]]
            if len(found) < req["count"]:
                correct = matched = False
                reasons.append(f"expected {classname}>={req['count']}, found {len(found)}")
            else:
                if "color" in req:
                    colors = self._color_classify(image, found, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reasons.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            f"{colors.count(req['color'])} {req['color']}"
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                        correct = matched = False
                        reasons.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found:
                            for target_obj in matched_groups[target_group]:
                                rels = relative_position(obj, target_obj)
                                if expected_rel not in rels:
                                    correct = matched = False
                                    reasons.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        f"{' and '.join(rels) if rels else 'none'} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found)
            else:
                matched_groups.append(None)

        for req in exclude:
            classname = req["class"]
            if len(detected.get(classname, [])) >= req["count"]:
                correct = False
                reasons.append(
                    f"expected {classname}<{req['count']}, found {len(detected[classname])}"
                )

        num_reqs = len(include) + len(exclude)
        num_failed = len(reasons) if not correct else 0
        return correct, "\n".join(reasons), num_reqs, num_failed

    # --- public ---

    def score(self, image: Image.Image, metadata: Dict) -> Dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        counting = (metadata.get("tag") == "counting")
        detected = self._detect(image, counting=counting)
        correct, reason, num_reqs, num_failed = self._evaluate(image, detected, metadata)
        fraction = 1.0 if num_reqs == 0 else max(0.0, (num_reqs - num_failed) / num_reqs)
        detected_summary = {k: {"count": len(v), "top_score": float(v[0][0][4]) if v else 0.0}
                             for k, v in detected.items()}
        return {
            "success": bool(correct),
            "correct": bool(correct),
            "reason": reason,
            "fraction": float(fraction),
            "num_reqs": int(num_reqs),
            "num_failed": int(num_failed),
            "category": metadata.get("tag"),
            "detected_summary": detected_summary,
            # Back-compat with OwlViT scorer
            "confidence": max([v["top_score"] for v in detected_summary.values()], default=0.0),
            "detected": detected_summary,
        }

    def score_batch(self, images: List[Image.Image], metadata: Dict) -> List[Dict[str, Any]]:
        return [self.score(im, metadata) for im in images]


# --- CLI (compat with original score_geneval.py: --image-dir + --metadata) ---

def _cli():
    import argparse
    import json
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--metadata", required=True, help="JSON string")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--counting-threshold", type=float, default=0.9)
    ap.add_argument("--max-objects", type=int, default=16)
    ap.add_argument("--nms-threshold", type=float, default=1.0)
    args = ap.parse_args()

    md = json.loads(args.metadata)
    scorer = NativeGenEvalScorer(
        device=args.device, threshold=args.threshold,
        counting_threshold=args.counting_threshold,
        max_objects=args.max_objects, nms_threshold=args.nms_threshold,
    )

    files = sorted(
        [f for f in os.listdir(args.image_dir) if re.match(r"\d+\.png", f)],
        key=lambda x: int(x.replace(".png", "")),
    )
    scores = []
    for f in files:
        img = Image.open(os.path.join(args.image_dir, f))
        r = scorer.score(img, md)
        scores.append(r["fraction"])
    print(json.dumps(scores))


if __name__ == "__main__":
    _cli()
