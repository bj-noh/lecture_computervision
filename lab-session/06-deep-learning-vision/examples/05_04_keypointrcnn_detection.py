"""
05_04_keypointrcnn_detection.py
────────────────────────────────────────────────────────────────────────────
Keypoint R-CNN 을 사용해 사람의 주요 관절 키포인트를 탐지합니다.

실행 예시:
    python 05_04_keypointrcnn_detection.py --image data/view.jpg

출력:
    outputs/05_04_keypointrcnn_detection.png
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
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

from output_naming import with_script_prefix


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Keypoint R-CNN person keypoint detection.")
    parser.add_argument("--image", type=Path, default=Path("data/view.jpg"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / with_script_prefix(__file__, "keypointrcnn_detection.png"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([tensor])[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(np.array(image))
    ax.axis("off")

    kept = 0
    for idx, score in enumerate(prediction["scores"].cpu().tolist()):
        if score < args.threshold:
            continue
        kept += 1
        box = prediction["boxes"][idx].cpu().numpy()
        keypoints = prediction["keypoints"][idx].cpu().numpy()

        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="#ef4444",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 6, f"person {score:.2f}", color="#ef4444", fontsize=10, weight="bold")

        for x, y, v in keypoints:
            if v < 1:
                continue
            ax.scatter(x, y, s=18, c="#06b6d4")

        for start, end in SKELETON:
            x1, y1, v1 = keypoints[start]
            x2, y2, v2 = keypoints[end]
            if v1 < 1 or v2 < 1:
                continue
            ax.plot([x1, x2], [y1, y2], color="#11b981", linewidth=2)

        print(f"person: {score:.4f}")

    if kept == 0:
        print("No people passed the threshold.")

    fig.tight_layout()
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
