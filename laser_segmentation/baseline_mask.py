"""Simple HSV-threshold laser mask, used as a fast non-learned baseline."""

import cv2
import numpy as np

_LOWER_HSV = (0, 100, 0)
_UPPER_HSV = (255, 255, 255)


def mask_simple(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _LOWER_HSV, _UPPER_HSV)


def mask_complex(image, row_start=10, row_end=1070):
    """Same HSV threshold, but blanks out rows outside [row_start, row_end)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    full_mask = cv2.inRange(hsv, _LOWER_HSV, _UPPER_HSV)

    mask = np.zeros(image.shape[:2])
    mask[row_start:row_end, :] = full_mask[row_start:row_end, :]
    return mask
