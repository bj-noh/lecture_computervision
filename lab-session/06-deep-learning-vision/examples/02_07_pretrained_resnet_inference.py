"""
02_07_pretrained_resnet_inference.py
────────────────────────────────────────────────────────────────────────────
ImageNet 으로 사전학습된 ResNet-18을 사용해 단일 이미지 분류를 수행합니다.

실행 예시:
    python 02_07_pretrained_resnet_inference.py --image data/dog1.jpg

출력:
    outputs/02_07_resnet_topk.png
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from output_naming import with_script_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ImageNet inference with ResNet-18.")
    parser.add_argument("--image", type=Path, default=Path("data/dog1.jpg"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / with_script_prefix(__file__, "resnet_topk.png"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    image = Image.open(args.image).convert("RGB")
    batch = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]

    values, indices = probs.topk(args.topk)
    labels = [categories[index] for index in indices.tolist()]
    scores = values.cpu().tolist()

    for label, score in zip(labels, scores):
        print(f"{label}: {score:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title(args.image.name)
    axes[0].axis("off")
    axes[1].barh(range(len(labels)), scores[::-1], color="#4f8cff")
    axes[1].set_yticks(range(len(labels)), labels[::-1])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Top Predictions")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
