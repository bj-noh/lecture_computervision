"""
05_03_maskrcnn_segmentation.py
────────────────────────────────────────────────────────────────────────────
Mask R-CNN 을 사용해 객체 탐지와 인스턴스 세그멘테이션을 수행합니다.

실행 예시:
    python 05_03_maskrcnn_segmentation.py --image data/view.jpg

출력:
    outputs/05_03_maskrcnn_segmentation.png
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

from output_naming import with_script_prefix


COLORS = ["#4f8cff", "#11b981", "#f97316", "#ef4444", "#8b5cf6", "#06b6d4"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mask R-CNN instance segmentation.")
    parser.add_argument("--image", type=Path, default=Path("data/view.jpg"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / with_script_prefix(__file__, "maskrcnn_segmentation.png"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = weights.meta["categories"]
    model = maskrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([tensor])[0]

    image_np = np.array(image).astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_np)
    ax.axis("off")

    kept = 0
    for idx, score in enumerate(prediction["scores"].cpu().tolist()):
        if score < args.threshold:
            continue
        kept += 1
        box = prediction["boxes"][idx].cpu().numpy()
        label_id = int(prediction["labels"][idx].cpu().item())
        label = categories[label_id]
        mask = prediction["masks"][idx, 0].cpu().numpy() > 0.5
        color = COLORS[idx % len(COLORS)]

        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgb = tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        overlay[mask] = (*rgb, 0.35)
        ax.imshow(overlay)
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 6, f"{label} {score:.2f}", color=color, fontsize=10, weight="bold")
        print(f"{label}: {score:.4f}")

    if kept == 0:
        print("No objects passed the threshold.")

    fig.tight_layout()
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
