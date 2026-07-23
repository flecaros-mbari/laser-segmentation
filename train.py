#!/usr/bin/env python3
"""Train the U-Net laser-segmentation model.

Example:
    python train.py \\
        --image-dir data/images --mask-dir data/masks \\
        --output-dir runs/highres_nonilluminated --test-name highres_nonilluminated
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from laser_segmentation.dataset import SegmentationDataset, paired_image_mask_paths
from laser_segmentation.model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True, help="Directory of input images")
    parser.add_argument("--mask-dir", required=True, help="Directory of ground-truth masks (same filenames as images)")
    parser.add_argument("--output-dir", required=True, help="Where to write the model, loss plot, and split file lists")
    parser.add_argument("--test-name", required=True, help="Name used as a prefix for all output files")

    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--num-classes", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.15, help="Recorded in metadata; used later at inference time")

    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of data held out for test+validation")
    parser.add_argument("--val-split", type=float, default=0.5, help="Fraction of the held-out data used for validation (rest is test)")

    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def split_dataset(image_paths, mask_paths, test_split, val_split, seed):
    train_images, holdout_images, train_masks, holdout_masks = train_test_split(
        image_paths, mask_paths, test_size=test_split, random_state=seed
    )
    test_images, val_images, test_masks, val_masks = train_test_split(
        holdout_images, holdout_masks, test_size=val_split, random_state=seed
    )
    return {
        "train": (train_images, train_masks),
        "test": (test_images, test_masks),
        "val": (val_images, val_masks),
    }


def write_path_list(paths, path):
    with open(path, "w") as f:
        f.write("\n".join(paths))


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.test_name)

    image_paths, mask_paths = paired_image_mask_paths(args.image_dir, args.mask_dir)
    splits = split_dataset(image_paths, mask_paths, args.test_split, args.val_split, args.seed)

    for split_name, (images, masks) in splits.items():
        write_path_list(images, f"{prefix}_{split_name}_images.txt")
        write_path_list(masks, f"{prefix}_{split_name}_masks.txt")

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    train_images, train_masks = splits["train"]
    test_images, test_masks = splits["test"]
    train_ds = SegmentationDataset(train_images, train_masks, transform)
    test_ds = SegmentationDataset(test_images, test_masks, transform)
    print(f"[INFO] found {len(train_ds)} training examples, {len(test_ds)} test examples")

    pin_memory = args.device == "cuda"
    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=pin_memory, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=args.batch_size, pin_memory=pin_memory, num_workers=args.num_workers
    )

    unet = UNet(
        num_classes=args.num_classes, out_size=(args.image_height, args.image_width)
    ).to(args.device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(unet.parameters(), args.lr)

    train_steps = max(len(train_ds) // args.batch_size, 1)
    test_steps = max(len(test_ds) // args.batch_size, 1)
    history = {"train_loss": [], "test_loss": []}

    print("[INFO] training the network...")
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        unet.train()
        total_train_loss = 0
        total_test_loss = 0

        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            pred = unet(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss

        with torch.no_grad():
            unet.eval()
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                pred = unet(x)
                total_test_loss += loss_fn(pred, y)

        avg_train_loss = (total_train_loss / train_steps).item()
        avg_test_loss = (total_test_loss / test_steps).item()
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)

        print(f"[INFO] EPOCH: {epoch + 1}/{args.epochs}")
        print(f"Train loss: {avg_train_loss:.6f}, Test loss: {avg_test_loss:.4f}")

    print(f"[INFO] total time taken to train the model: {time.time() - start_time:.2f}s")

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(f"{prefix}.png")

    torch.save(unet.state_dict(), f"{prefix}.pt")
    with open(f"{prefix}_meta.json", "w") as f:
        json.dump(
            {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "num_classes": args.num_classes,
                "threshold": args.threshold,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "test_split": args.test_split,
            },
            f,
            indent=2,
        )

    print(f"[INFO] saved model, loss plot, metadata, and split lists under {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
