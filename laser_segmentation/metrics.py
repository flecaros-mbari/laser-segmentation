"""Pixel- and point-based comparison metrics between a ground-truth mask and
a predicted mask.

This consolidates three near-identical blocks from the original notebook
(one per comparison it ran: predictions vs. MOLA masks, predictions vs. test
masks, HSV-threshold vs. test masks) into a single implementation, since all
three were computing the same metrics and had quietly drifted out of sync
with each other for edge cases.
"""

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as scipy_distance
from scipy.spatial.distance import directed_hausdorff


def confusion_matrix(real_mask_path, predicted_mask_path):
    """Pixel-level confusion matrix between two binary (0/255) masks.

    Returns (true_white, true_black, false_negative, false_positive), i.e.
    (both white, both black, real white but predicted black, real black but
    predicted white). Returns all zeros if the images differ in shape.
    """
    real = cv2.imread(real_mask_path, 0)
    predicted = cv2.imread(predicted_mask_path, 0)

    if real.shape != predicted.shape:
        return 0, 0, 0, 0

    real_white = real == 255
    predicted_white = predicted == 255

    true_white = int(np.sum(real_white & predicted_white))
    true_black = int(np.sum(~real_white & ~predicted_white))
    false_negative = int(np.sum(real_white & ~predicted_white))
    false_positive = int(np.sum(~real_white & predicted_white))

    return true_white, true_black, false_negative, false_positive


def one_line_pixels(mask_original_path, mask_predicted_path):
    """Average white-pixel row per column, for each mask.

    Returns two lists of (column, row) pairs (row is NaN where a column has
    no white pixels).
    """
    mask_original = cv2.imread(mask_original_path, 0)
    mask_predicted = cv2.imread(mask_predicted_path, 0)

    points_original = []
    points_predicted = []
    for col in range(mask_original.shape[1]):
        white_original = np.where(mask_original[:, col] > 0)[0]
        white_predicted = np.where(mask_predicted[:, col] > 0)[0]

        avg_original = round(np.mean(white_original)) if len(white_original) > 0 else np.nan
        avg_predicted = round(np.mean(white_predicted)) if len(white_predicted) > 0 else np.nan

        points_original.append((col, avg_original))
        points_predicted.append((col, avg_predicted))

    return points_original, points_predicted


def column_distance(real_points, predicted_points):
    """Per-column squared-distance stats between two one_line_pixels() outputs.

    Returns (rmse over comparable columns, sqrt of the summed squared
    distance, percentage of columns where either side has no point).
    """
    total_squared_diff = 0.0
    missing = 0
    for (real_col, real_row), (pred_col, pred_row) in zip(real_points, predicted_points):
        diff = real_row - pred_row
        if np.isnan(diff):
            missing += 1
        else:
            total_squared_diff += diff**2

    n = len(real_points)
    comparable = n - missing
    rmse = np.sqrt(total_squared_diff) / comparable if comparable else float("nan")

    return rmse, np.sqrt(total_squared_diff), missing / n * 100


def compare_images(img1, img2):
    """Manhattan norm and zero norm between two same-shaped float images."""
    diff = img1 - img2
    manhattan_norm = np.sum(np.abs(diff))
    zero_norm = np.count_nonzero(diff)
    return manhattan_norm, zero_norm


def norms_distances(mask_path_1, mask_path_2):
    mask1 = cv2.imread(mask_path_1, 0).astype(float)
    mask2 = cv2.imread(mask_path_2, 0).astype(float)

    manhattan_norm, zero_norm = compare_images(mask1, mask2)
    return manhattan_norm, manhattan_norm / mask1.size, zero_norm, zero_norm / mask1.size


def point_distance(points1, points2):
    """Mean nearest-neighbor distance from points1 to points2 (NaNs dropped)."""
    points1 = np.array(points1, dtype=float)
    points2 = np.array(points2, dtype=float)

    points1 = points1[~np.isnan(points1).any(axis=1)]
    points2 = points2[~np.isnan(points2).any(axis=1)]

    if len(points1) == 0 or len(points2) == 0:
        return np.nan

    pairwise = scipy_distance.cdist(points1, points2, "euclidean")
    return pairwise.min(axis=1).mean()


def directed_hausdorff_modified(points1, points2):
    points1 = np.array(points1, dtype=float)
    points2 = np.array(points2, dtype=float)

    points1 = points1[~np.isnan(points1).any(axis=1)]
    points2 = points2[~np.isnan(points2).any(axis=1)]

    d_forward = directed_hausdorff(points1, points2)[0]
    d_backward = directed_hausdorff(points2, points1)[0]
    return max(d_forward, d_backward)


def _precision_recall_f1(true_white, true_black, false_negative, false_positive):
    if true_white + false_positive != 0:
        precision = true_white * 100 / (true_white + false_positive)
    else:
        precision = 100 if false_negative == 0 else 0

    if true_white + false_negative != 0:
        recall = true_white * 100 / (true_white + false_negative)
    else:
        recall = 100 if false_positive == 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1


def evaluate_predictions(original_mask_paths, predicted_mask_paths):
    """Compute per-image and averaged comparison metrics.

    original_mask_paths and predicted_mask_paths must be order-matched
    (same image, ground truth vs. prediction). Returns a DataFrame with one
    row per image plus a leading "average" row.
    """
    rows = []
    for original_path, predicted_path in zip(original_mask_paths, predicted_mask_paths):
        true_white, true_black, false_negative, false_positive = confusion_matrix(
            original_path, predicted_path
        )
        manhattan_norm, manhattan_norm_pp, zero_norm, zero_norm_pp = norms_distances(
            original_path, predicted_path
        )
        real_points, predicted_points = one_line_pixels(original_path, predicted_path)
        rmse, euclidean_distance, missing_pct = column_distance(real_points, predicted_points)
        precision, recall, f1 = _precision_recall_f1(
            true_white, true_black, false_negative, false_positive
        )

        total = true_white + true_black + false_negative + false_positive
        accuracy = (true_white + true_black) * 100 / total if total else 0

        rows.append(
            {
                "original_path": original_path,
                "predicted_path": predicted_path,
                "true_white": true_white,
                "true_black": true_black,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "rmse": rmse,
                "euclidean_distance": euclidean_distance,
                "missing_pct": missing_pct,
                "manhattan_norm": manhattan_norm,
                "manhattan_norm_pp": manhattan_norm_pp,
                "zero_norm": zero_norm,
                "zero_norm_pp": zero_norm_pp,
                "point_distance_fwd": point_distance(real_points, predicted_points),
                "point_distance_bwd": point_distance(predicted_points, real_points),
                "hausdorff": directed_hausdorff_modified(real_points, predicted_points),
            }
        )

    df = pd.DataFrame(rows)
    numeric_cols = df.columns.drop(["original_path", "predicted_path"])
    average_row = {"original_path": "-", "predicted_path": "-", **df[numeric_cols].mean().to_dict()}
    return pd.concat([pd.DataFrame([average_row]), df], ignore_index=True)
