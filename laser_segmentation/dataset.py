"""Paired image/mask dataset for training and evaluation."""

import os

import cv2
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(directory):
    """Recursively list image files under a directory, sorted by path."""
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                matches.append(os.path.join(root, filename))
    return sorted(matches)


def paired_image_mask_paths(image_dir, mask_dir):
    """Pair up images and masks that share the same filename.

    Returns two equal-length, order-matched lists. Images without a
    matching mask filename (or vice versa) are dropped.
    """
    image_paths = list_images(image_dir)
    mask_by_name = {os.path.basename(p): p for p in list_images(mask_dir)}

    pairs = [
        (image_path, mask_by_name[os.path.basename(image_path)])
        for image_path in image_paths
        if os.path.basename(image_path) in mask_by_name
    ]
    if not pairs:
        raise ValueError(
            f"No matching image/mask filename pairs between {image_dir!r} and {mask_dir!r}"
        )

    images, masks = zip(*pairs)
    return list(images), list(masks)


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask
