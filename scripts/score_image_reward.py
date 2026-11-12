#!/usr/bin/env python3
"""Standalone ImageReward scoring script.

Runs in a separate venv ($PROXYSEARCH_IR_PYTHON) because ImageReward requires
transformers 4.x which is incompatible with our main venv (transformers 5.x).

Usage:
    $PROXYSEARCH_IR_PYTHON scripts/score_image_reward.py \
        --image-dir path/to/images \
        --prompt "a red apple on a table" \
        --device cuda

Expects numbered PNG files in image-dir: 0.png, 1.png, 2.png, ...
Outputs JSON array of scores to stdout.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory containing numbered .png files")
    parser.add_argument("--prompt", required=True, help="Text prompt for scoring")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    args = parser.parse_args()

    import ImageReward as ir
    model = ir.load("ImageReward-v1.0", device=args.device)

    # Find all numbered images
    image_files = sorted(
        [f for f in os.listdir(args.image_dir) if f.endswith(".png")],
        key=lambda x: int(x.split(".")[0]),
    )

    scores = []
    for fname in image_files:
        img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
        score = model.score(args.prompt, img)
        scores.append(float(score))

    # Output as JSON to stdout
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
