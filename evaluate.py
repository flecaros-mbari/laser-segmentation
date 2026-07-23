#!/usr/bin/env python3
"""Compare predicted masks against ground truth and write metrics to Excel.

Run once per comparison you want in the report (e.g. once for the U-Net
predictions, once for the HSV baseline), pointing --sheet-name at a
different sheet each time; rows accumulate in the same workbook.

Example:
    python evaluate.py --mask-dir runs/highres/mola_predictions/mask \\
        --predicted-dir runs/highres/mola_predictions/predicted_0.15 \\
        --output-xlsx runs/highres/results.xlsx --sheet-name "unet_0.15"
"""

import argparse
import os

import pandas as pd

from laser_segmentation.dataset import list_images
from laser_segmentation.metrics import evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask-dir", required=True, help="Directory of ground-truth masks")
    parser.add_argument("--predicted-dir", required=True, help="Directory of predicted masks (same filenames as masks)")
    parser.add_argument("--output-xlsx", required=True)
    parser.add_argument("--sheet-name", required=True)
    return parser.parse_args()


def paired_mask_paths(mask_dir, predicted_dir):
    mask_paths = list_images(mask_dir)
    predicted_by_name = {os.path.basename(p): p for p in list_images(predicted_dir)}

    pairs = [
        (mask_path, predicted_by_name[os.path.basename(mask_path)])
        for mask_path in mask_paths
        if os.path.basename(mask_path) in predicted_by_name
    ]
    if not pairs:
        raise ValueError(f"No matching filenames between {mask_dir!r} and {predicted_dir!r}")

    masks, predictions = zip(*pairs)
    return list(masks), list(predictions)


def run(args):
    mask_paths, predicted_paths = paired_mask_paths(args.mask_dir, args.predicted_dir)
    print(f"[INFO] comparing {len(mask_paths)} mask/prediction pairs...")

    df = evaluate_predictions(mask_paths, predicted_paths)

    write_mode = "a" if os.path.exists(args.output_xlsx) else "w"
    kwargs = {"if_sheet_exists": "replace"} if write_mode == "a" else {}
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl", mode=write_mode, **kwargs) as writer:
        df.to_excel(writer, sheet_name=args.sheet_name, index=False)

    print(f"[INFO] wrote sheet {args.sheet_name!r} to {args.output_xlsx}")


if __name__ == "__main__":
    run(parse_args())
