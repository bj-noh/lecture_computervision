"""
02_08_vit_flowers102.py
────────────────────────────────────────────────────────────────────────────
Vision Transformer (ViT-B/16)을 사용해 Oxford 102 Flowers 데이터셋을 분류합니다.

ViT (Dosovitskiy et al., 2020) 핵심 아이디어:
  - 이미지를 16×16 패치로 나눈 뒤 각 패치를 토큰처럼 다룸
  - CNN 대신 Transformer Encoder 로 전역 문맥을 학습
  - 대규모 사전학습 후 전이학습에서 강한 성능을 보임

Oxford 102 Flowers 데이터셋:
  - 102개 꽃 카테고리
  - train 1,020장 / val 1,020장 / test 6,149장
  - train+val 을 합쳐 2,040장으로 학습

실행 예시:
    python 02_08_vit_flowers102.py --epochs 12 --pretrained
    python 02_08_vit_flowers102.py --epochs 12 --pretrained --freeze-features

출력:
    outputs/02_08_vit_flowers102_vit_b16_{mode}_history.png
    outputs/02_08_vit_flowers102_vit_b16_{mode}_best.pt
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

from output_naming import with_script_prefix


def build_vit(
    num_classes: int = 102,
    pretrained: bool = False,
    freeze_features: bool = False,
) -> nn.Module:
    """ViT-B/16을 Flowers-102 분류용으로 구성한다."""
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)

    if pretrained:
        print("✔ ViT-B/16 ImageNet pre-trained weights loaded")
    else:
        print("✔ ViT-B/16 initialized from scratch")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    print(f"✔ classification head replaced: Linear({in_features}, 1000) -> Linear({in_features}, {num_classes})")

    if freeze_features:
        for name, param in model.named_parameters():
            if not name.startswith("heads"):
                param.requires_grad = False
        print("✔ transformer encoder frozen (classification head only)")

    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    """평균 손실과 Top-1 정확도를 반환한다."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
) -> tuple[float, float]:
    """한 에폭 학습 후 평균 손실과 정확도를 반환한다."""
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def plot_history(history: dict[str, list[float]], epochs: int, title: str, output_path: Path) -> None:
    """손실/정확도 곡선을 저장한다."""
    x = range(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(x, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(x, history["test_loss"], label="Test Loss", color="#2980b9", linewidth=2, linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, [v * 100 for v in history["train_acc"]], label="Train Acc", color="#e74c3c", linewidth=2)
    ax2.plot(x, [v * 100 for v in history["test_acc"]], label="Test Acc", color="#2980b9", linewidth=2, linestyle="--")
    ax2.set_title("Top-1 Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved history plot: {output_path}")


def output_tag(pretrained: bool, freeze_features: bool) -> str:
    if not pretrained:
        return "scratch"
    if freeze_features:
        return "pretrained_frozen"
    return "pretrained"


def mode_labels(pretrained: bool, freeze_features: bool) -> tuple[str, str]:
    if not pretrained:
        return "Scratch", "Scratch"
    if freeze_features:
        return "Pretrained (Frozen Backbone)", "Pretrained (Frozen Backbone)"
    return "Pretrained", "Pretrained"


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT-B/16 on Oxford Flowers-102.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pre-trained weights")
    parser.add_argument("--freeze-features", action="store_true", help="Train only the classification head")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_mode, plot_mode = mode_labels(args.pretrained, args.freeze_features)
    run_tag = output_tag(args.pretrained, args.freeze_features)

    print(f"Device:     {device}")
    print("Architecture: ViT-B/16")
    print(f"Mode:       {display_mode}")
    print("Dataset:    Oxford 102 Flowers")

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.Flowers102(root=args.data_root, split="train", download=True, transform=train_transform)
    val_set = datasets.Flowers102(root=args.data_root, split="val", download=True, transform=test_transform)
    test_set = datasets.Flowers102(root=args.data_root, split="test", download=True, transform=test_transform)
    trainval_set = ConcatDataset([train_set, val_set])

    train_loader = DataLoader(trainval_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(trainval_set):,}  /  Test samples: {len(test_set):,}")

    model = build_vit(
        num_classes=102,
        pretrained=args.pretrained,
        freeze_features=args.freeze_features,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total / {trainable_params:,} trainable")

    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    lr = args.lr * 10 if args.pretrained and args.freeze_features else args.lr
    optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=amp_enabled)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_test_acc = 0.0
    ckpt_path = args.output_dir / with_script_prefix(__file__, f"vit_flowers102_vit_b16_{run_tag}_best.pt")

    print(f"{'Epoch':>6} {'LR':>10} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9}")
    print("-" * 66)

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, amp_enabled)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, amp_enabled)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        best_flag = " ★" if test_acc > best_test_acc else ""
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"{epoch:>6d} {current_lr:>10.2e} {train_loss:>11.4f} "
            f"{train_acc * 100:>9.2f}% {test_loss:>10.4f} {test_acc * 100:>8.2f}%{best_flag}"
        )

    best_epoch = history["test_acc"].index(max(history["test_acc"])) + 1
    print("\n" + "=" * 66)
    print("Final Summary (ViT-B/16 on Flowers-102)")
    print("=" * 66)
    print(f"  Best test accuracy: {best_test_acc * 100:.2f}%  (epoch {best_epoch})")
    print(f"  Final train accuracy: {history['train_acc'][-1] * 100:.2f}%")
    print(f"  Final test accuracy:  {history['test_acc'][-1] * 100:.2f}%")
    print(f"  Best checkpoint:      {ckpt_path}")

    plot_history(
        history=history,
        epochs=args.epochs,
        title=f"ViT-B/16 on Flowers-102 - {plot_mode}",
        output_path=args.output_dir / with_script_prefix(__file__, f"vit_flowers102_vit_b16_{run_tag}_history.png"),
    )


if __name__ == "__main__":
    main()
