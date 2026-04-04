from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an augmentation gallery.")
    parser.add_argument("--image", type=Path, default=Path("data/cat.jpg"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/augmentation_grid.png"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ])

    cols = 4
    rows = math.ceil((args.num_samples + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[0].axis("off")

    for idx in range(1, args.num_samples + 1):
        augmented = augment(image)
        axes[idx].imshow(augmented)
        axes[idx].set_title(f"aug {idx}")
        axes[idx].axis("off")

    for idx in range(args.num_samples + 1, rows * cols):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
