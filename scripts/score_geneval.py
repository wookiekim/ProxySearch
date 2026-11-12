#!/usr/bin/env python3
"""
Standalone GenEval scorer. Runs in the GenEval venv (see INSTALL.md).

Loads Mask2Former + CLIP once, scores all images in --image-dir against
the provided --metadata JSON. Outputs JSON array of per-image scores to stdout.

Usage:
    $PROXYSEARCH_GENEVAL_PYTHON scripts/score_geneval.py \
        --image-dir path/to/images \
        --metadata '{"tag":"single_object","include":[{"class":"cow","count":1}],"prompt":"a photo of a cow"}' \
        --device cuda
"""

import argparse
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageOps
import torch

# GenEval evaluation code imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "geneval", "evaluation"))


def parse_args():
    parser = argparse.ArgumentParser(description="Score images with GenEval")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory with numbered PNGs (0.png, 1.png, ...)")
    parser.add_argument("--metadata", type=str, required=True,
                        help="JSON string with GenEval metadata for this prompt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-path", type=str, default="benchmarks/geneval/models/",
                        help="Path to Mask2Former checkpoint directory")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--counting-threshold", type=float, default=0.9)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--nms-threshold", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = json.loads(args.metadata)

    # Import mmdet and load models
    import mmdet
    from mmdet.apis import inference_detector, init_detector

    model_config = os.path.join(
        os.path.dirname(mmdet.__file__),
        "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    )

    # Find checkpoint
    ckpt_path = None
    for f in os.listdir(args.model_path):
        if f.endswith(".pth"):
            ckpt_path = os.path.join(args.model_path, f)
            break
    if ckpt_path is None:
        print("ERROR: No .pth checkpoint found in " + args.model_path, file=sys.stderr)
        sys.exit(1)

    object_detector = init_detector(model_config, ckpt_path, device=args.device)
    # mmdet 2.x: CLASSES is a tuple of strings
    if hasattr(object_detector, 'CLASSES'):
        cls = object_detector.CLASSES
        if isinstance(cls[0], str):
            classnames = cls
        else:
            classnames = [c['name'] for c in cls]
    else:
        from mmdet.datasets import CocoDataset
        classnames = CocoDataset.CLASSES

    # Import evaluate functions from GenEval
    from evaluate_images import evaluate, color_classification, relative_position, compute_iou, COLORS

    threshold = args.threshold
    counting_threshold = args.counting_threshold
    max_objects = args.max_objects
    nms_threshold = args.nms_threshold

    # Find all numbered images
    image_files = []
    for f in os.listdir(args.image_dir):
        if re.match(r"\d+\.png", f):
            image_files.append(f)
    image_files.sort(key=lambda x: int(x.replace(".png", "")))

    scores = []
    for fname in image_files:
        filepath = os.path.join(args.image_dir, fname)
        # Run detection
        result = inference_detector(object_detector, filepath)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(Image.open(filepath))

        # Parse detections
        conf_threshold = threshold if metadata['tag'] != "counting" else counting_threshold
        detected = {}
        for index, classname in enumerate(classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > conf_threshold]
            ordering = ordering[:max_objects].tolist()
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append(
                    (bbox[index][max_obj], None if segm is None else segm[index][max_obj])
                )
                ordering = [
                    obj for obj in ordering
                    if nms_threshold == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < nms_threshold
                ]
            if not detected[classname]:
                del detected[classname]

        is_correct, reason = evaluate(image, detected, metadata)
        # Compute fractional score: fraction of requirements met
        num_reqs = len(metadata.get('include', [])) + len(metadata.get('exclude', []))
        if num_reqs > 0 and reason:
            num_failed = len(reason.strip().split("\n")) if reason.strip() else 0
            frac = max(0.0, (num_reqs - num_failed) / num_reqs)
        elif num_reqs > 0:
            frac = 1.0
        else:
            frac = 1.0 if is_correct else 0.0
        scores.append(frac)

    # Output JSON array to stdout
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
