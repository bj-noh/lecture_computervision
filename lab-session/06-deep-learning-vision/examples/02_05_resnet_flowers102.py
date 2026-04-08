"""
02_05_resnet_flowers102.py
────────────────────────────────────────────────────────────────────────────
torchvision.models 의 ResNet 계열(18/34/50/101)을 사용해
Oxford 102 Flowers 데이터셋을 분류합니다.

ResNet (He et al., 2015) 핵심 아이디어 — 잔차 연결(Residual Connection):
  기존 CNN: H(x) = F(x)         (입력 x를 직접 변환)
  ResNet:   H(x) = F(x) + x     (잔차 F(x)만 학습, x는 shortcut으로 통과)

  → 깊은 네트워크에서 발생하던 기울기 소실(Gradient Vanishing) 문제 해결
  → VGGNet(19층)보다 훨씬 깊은 100층 이상 학습 가능

Basic Block (ResNet-18/34):       Bottleneck Block (ResNet-50/101/152):
  3×3 Conv → BN → ReLU             1×1 Conv → BN → ReLU   (채널 축소)
  3×3 Conv → BN                    3×3 Conv → BN → ReLU
  + shortcut (identity / proj.)     1×1 Conv → BN           (채널 복원)
  → ReLU                            + shortcut
                                    → ReLU

Oxford 102 Flowers 데이터셋:
  - 102개 꽃 카테고리 (장미, 해바라기, 튤립 등 세분화된 클래스)
  - 훈련: 1,020장 / 검증: 1,020장 / 테스트: 6,149장
  - 이미지 크기: 다양 → 224×224 리사이즈
  - 클래스당 샘플 수가 적어 전이학습 효과를 잘 확인할 수 있는 데이터셋

실행 예시:
    python 02_05_resnet_flowers102.py --arch resnet50 --epochs 20 --pretrained
    python 02_05_resnet_flowers102.py --arch resnet18 --epochs 30

출력:
    outputs/resnet_flowers102_{arch}_history.png
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights,
    ResNet50_Weights, ResNet101_Weights,
)

from output_naming import with_script_prefix


# ══════════════════════════════════════════════════════════════════════════
# 지원 아키텍처 설정
# ══════════════════════════════════════════════════════════════════════════
ARCH_CONFIG: dict[str, tuple] = {
    "resnet18":  (models.resnet18,  ResNet18_Weights.DEFAULT),
    "resnet34":  (models.resnet34,  ResNet34_Weights.DEFAULT),
    "resnet50":  (models.resnet50,  ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, ResNet101_Weights.DEFAULT),
}

# ResNet 계열별 블록 구성 정보 (교육용 출력)
ARCH_INFO: dict[str, str] = {
    "resnet18":  "BasicBlock  × [2,2,2,2] → 18층",
    "resnet34":  "BasicBlock  × [3,4,6,3] → 34층",
    "resnet50":  "Bottleneck  × [3,4,6,3] → 50층",
    "resnet101": "Bottleneck  × [3,4,23,3] → 101층",
}


# ══════════════════════════════════════════════════════════════════════════
# 모델 빌드
# ══════════════════════════════════════════════════════════════════════════
def build_resnet(
    arch: str,
    num_classes: int = 102,
    pretrained: bool = False,
    freeze_features: bool = False,
) -> nn.Module:
    """
    torchvision 내장 ResNet을 Flowers-102 (102클래스)용으로 수정하여 반환.

    ResNet의 fc 레이어 교체:
      ResNet 출력: model.fc = Linear(512/2048, 1000)
        - ResNet-18/34: 512차원 (BasicBlock)
        - ResNet-50/101: 2048차원 (Bottleneck)
      → Linear(512 or 2048, num_classes) 로 교체

    Args:
        arch:            'resnet18' | 'resnet34' | 'resnet50' | 'resnet101'
        num_classes:     출력 클래스 수 (Flowers102 = 102)
        pretrained:      ImageNet 사전학습 가중치 사용 여부
        freeze_features: True면 fc 이전 레이어 파라미터 고정
    """
    if arch not in ARCH_CONFIG:
        raise ValueError(f"지원하지 않는 아키텍처: {arch}. {list(ARCH_CONFIG.keys())} 중 선택하세요.")

    builder, weights = ARCH_CONFIG[arch]

    if pretrained:
        model = builder(weights=weights)
        print(f"✔ {arch.upper()} ImageNet 사전학습 가중치 로드 완료")
        print(f"  블록 구성: {ARCH_INFO[arch]}")
    else:
        model = builder(weights=None)
        print(f"✔ {arch.upper()} 가중치 없이 생성 (scratch)")
        print(f"  블록 구성: {ARCH_INFO[arch]}")

    # ── 마지막 FC 레이어 교체: 1000 → num_classes ────────────────────────
    # ResNet은 VGG와 달리 마지막 레이어가 model.fc 하나 뿐 (Global Avg Pool 후)
    in_features = model.fc.in_features     # resnet18/34: 512 / resnet50/101: 2048
    model.fc    = nn.Linear(in_features, num_classes)
    print(f"✔ model.fc 교체: Linear({in_features}, 1000) → Linear({in_features}, {num_classes})")

    # ── 전이학습: fc 이전 레이어 파라미터 고정 ───────────────────────────
    if freeze_features:
        # fc 레이어를 제외한 모든 레이어 고정
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        print("✔ fc 이전 레이어 파라미터 고정 (fc만 학습)")

    return model


# ══════════════════════════════════════════════════════════════════════════
# 학습·평가 함수
# ══════════════════════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """테스트/검증 세트 평균 손실과 Top-1 정확도를 반환."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits         = model(images)
            total_loss    += loss_fn(logits, labels).item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count   += images.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """한 에폭 학습 후 (평균 훈련 손실, 훈련 정확도)를 반환."""
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count   += images.size(0)

    return total_loss / total_count, total_correct / total_count


# ══════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════
def plot_history(history: dict[str, list[float]], epochs: int, title: str, output_path: Path) -> None:
    """훈련/테스트 손실·정확도 곡선을 저장."""
    x = range(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(x, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(x, history["test_loss"],  label="Test Loss",  color="#2980b9", linewidth=2, linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#e74c3c", linewidth=2)
    ax2.plot(x, [a * 100 for a in history["test_acc"]],  label="Test Acc",  color="#2980b9", linewidth=2, linestyle="--")
    ax2.set_title("Top-1 Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"그래프 저장 완료: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── 인자 파싱 ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="ResNet on Oxford Flowers-102 (torchvision).")
    parser.add_argument("--arch",            type=str,   default="resnet50",
                        choices=list(ARCH_CONFIG.keys()),
                        help="ResNet 변형 선택 (resnet18 | resnet34 | resnet50 | resnet101)")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data-root",       type=Path,  default=Path("data"))
    parser.add_argument("--output-dir",      type=Path,  default=Path("outputs"))
    parser.add_argument("--pretrained",      action="store_true",
                        help="ImageNet 사전학습 가중치 사용 (전이학습)")
    parser.add_argument("--freeze-features", action="store_true",
                        help="fc 이전 레이어 파라미터 고정 (--pretrained와 함께 사용)")
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "전이학습 (Pretrained)" if args.pretrained else "Scratch"
    print(f"사용 장치: {device}")
    print(f"아키텍처:  {args.arch.upper()}")
    print(f"모드:      {mode}")
    print(f"데이터셋:  Oxford 102 Flowers (102 classes)\n")

    # ── 데이터 전처리 ─────────────────────────────────────────────────────
    # Flowers102는 이미지 크기가 제각각 → CenterCrop으로 통일

    # 사전학습 여부에 따라 정규화 통계 선택
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),   # ImageNet 통계 (pretrained 여부 무관하게 사용)
        std =(0.229, 0.224, 0.225),   # → 사전학습 시 필수, scratch 시에도 관행적으로 사용
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),      # 랜덤 크롭 후 224×224 리사이즈 (강력한 증강)
        transforms.RandomHorizontalFlip(),       # 좌우 반전
        transforms.ColorJitter(                  # 색상 변형 (꽃 데이터셋에 효과적)
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),          # 먼저 256으로 리사이즈
        transforms.CenterCrop(224),      # 중앙 224×224 크롭 (평가 시 표준 방식)
        transforms.ToTensor(),
        normalize,
    ])

    # Flowers102: split='train'(1020장), 'val'(1020장), 'test'(6149장)
    train_set = datasets.Flowers102(root=args.data_root, split="train", download=True, transform=train_transform)
    val_set   = datasets.Flowers102(root=args.data_root, split="val",   download=True, transform=test_transform)
    test_set  = datasets.Flowers102(root=args.data_root, split="test",  download=True, transform=test_transform)

    # 훈련 샘플이 적으므로 train+val 합쳐서 학습 (일반적인 Flowers102 사용법)
    from torch.utils.data import ConcatDataset
    trainval_set = ConcatDataset([train_set, val_set])  # 1020 + 1020 = 2040장

    train_loader = DataLoader(trainval_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,     batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"훈련 샘플(train+val): {len(trainval_set):,}  /  테스트 샘플: {len(test_set):,}")
    print(f"※ 클래스당 훈련 샘플 수가 매우 적음 → 전이학습(--pretrained)이 효과적\n")

    # ── 모델 빌드 ─────────────────────────────────────────────────────────
    model = build_resnet(
        arch            = args.arch,
        num_classes     = 102,
        pretrained      = args.pretrained,
        freeze_features = args.freeze_features,
    )
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n전체 파라미터: {total:,}  /  학습 가능: {trainable:,}\n")

    # ── 옵티마이저·스케줄러·손실 함수 ────────────────────────────────────
    # 학습 가능한 파라미터만 옵티마이저에 전달
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.pretrained and args.freeze_features:
        # fc만 학습할 때는 더 큰 lr 사용
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr * 10)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)

    # CosineAnnealingLR: 학습률을 부드럽게 감소시켜 수렴 안정화
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)   # Label Smoothing: 102클래스 과적합 억제

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }

    print(f"{'Epoch':>6} {'LR':>8} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9}")
    print("-" * 62)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss,  test_acc  = evaluate(model, test_loader, loss_fn, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        best_flag = " ★" if test_acc > best_test_acc else ""
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(
            f"{epoch:>6d} {current_lr:>8.2e} {train_loss:>11.4f} "
            f"{train_acc*100:>9.2f}% {test_loss:>10.4f} {test_acc*100:>8.2f}%{best_flag}"
        )

    # ── 최종 결과 요약 ────────────────────────────────────────────────────
    best_epoch = history["test_acc"].index(max(history["test_acc"])) + 1
    print("\n" + "=" * 62)
    print(f"최종 결과 요약 ({args.arch.upper()} on Flowers-102)")
    print("=" * 62)
    print(f"  최고 테스트 정확도: {best_test_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  최종 훈련 정확도:   {history['train_acc'][-1]*100:.2f}%")
    print(f"  최종 테스트 정확도: {history['test_acc'][-1]*100:.2f}%")

    # ── 그래프 저장 ───────────────────────────────────────────────────────
    plot_history(
        history     = history,
        epochs      = args.epochs,
        title       = f"{args.arch.upper()} on Flowers-102 — {mode}",
        output_path = args.output_dir / with_script_prefix(__file__, f"resnet_flowers102_{args.arch}_history.png"),
    )


if __name__ == "__main__":
    main()
