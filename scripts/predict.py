#!/usr/bin/env python3
"""Run a trained U-Net over a directory of images and save its predictions.

Ground truth is optional: pass --mask-dir to also copy the matching ground
truth mask into the output directory (for later comparison with evaluate.py);
omit it to just run inference (e.g. on unlabeled field data).

Example:
    python scripts/predict.py --model-path runs/highres/highres.pt \\
        --image-dir data/mola_images --output-dir runs/highres/mola_predictions
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from laser_segmentation.baseline_mask import mask_complex
from laser_segmentation.dataset import list_images
from laser_segmentation.model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Path to a .pt state_dict saved by train.py")
    parser.add_argument("--meta-path", help="Path to the _meta.json saved by train.py (defaults to <model-path without .pt>_meta.json)")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", help="Optional ground-truth mask directory (same filenames as images)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, help="Overrides the threshold recorded in metadata")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_meta(args):
    meta_path = args.meta_path or f"{os.path.splitext(args.model_path)[0]}_meta.json"
    with open(meta_path) as f:
        return json.load(f)


def predict_image(unet, image_path, threshold, device):
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    tensor = torch.from_numpy(np.transpose(scaled, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(unet(tensor)).squeeze().cpu().numpy()

    pred_mask = ((pred > threshold) * 255).astype(np.uint8)
    baseline_mask = mask_complex(image).astype(np.uint8)
    return image, pred_mask, baseline_mask


def run(args):
    meta = load_meta(args)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]

    unet = UNet(
        num_classes=meta["num_classes"], out_size=(meta["image_height"], meta["image_width"])
    ).to(args.device)
    unet.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))
    unet.eval()

    dirs = {
        "original": os.path.join(args.output_dir, "original"),
        "predicted": os.path.join(args.output_dir, f"predicted_{threshold}"),
        "baseline": os.path.join(args.output_dir, "baseline_threshold"),
    }
    if args.mask_dir:
        dirs["mask"] = os.path.join(args.output_dir, "mask")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    image_paths = list_images(args.image_dir)
    print(f"[INFO] running inference on {len(image_paths)} images...")

    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        image, pred_mask, baseline_mask = predict_image(unet, image_path, threshold, args.device)

        cv2.imwrite(os.path.join(dirs["original"], name + ".png"), image)
        cv2.imwrite(os.path.join(dirs["predicted"], name + ".png"), pred_mask)
        cv2.imwrite(os.path.join(dirs["baseline"], name + ".png"), baseline_mask)

        if args.mask_dir:
            mask_path = os.path.join(args.mask_dir, os.path.basename(image_path))
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path)
                cv2.imwrite(os.path.join(dirs["mask"], name + ".png"), gt_mask)

    print(f"[INFO] wrote predictions to {args.output_dir}")


if __name__ == "__main__":
    run(parse_args())
