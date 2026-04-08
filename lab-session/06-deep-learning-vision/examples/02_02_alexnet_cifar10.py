"""
02_02_alexnet_cifar10.py
────────────────────────────────────────────────────────────────────────────
torchvision.models.alexnet 을 사용해 CIFAR-10을 분류합니다.

두 가지 모드를 지원합니다:
  1. scratch   : ImageNet 사전학습 없이 처음부터 학습 (--pretrained 생략)
  2. finetune  : ImageNet 사전학습 가중치 로드 후 미세조정 (--pretrained)

AlexNet 원본은 224×224 입력을 요구하므로 CIFAR-10 이미지(32×32)를
Resize(224)로 업스케일합니다.

실행 예시:
    # 처음부터 학습
    python 02_02_alexnet_cifar10.py --epochs 10

    # 전이학습 (사전학습 가중치 사용, 마지막 FC만 학습)
    python 02_02_alexnet_cifar10.py --epochs 10 --pretrained --freeze-features

출력:
    outputs/alexnet_cifar10_history.png  ← 손실·정확도 곡선
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
from torchvision.models import AlexNet_Weights

from output_naming import with_script_prefix


# ══════════════════════════════════════════════════════════════════════════
# 모델 준비
# ══════════════════════════════════════════════════════════════════════════
def build_alexnet(num_classes: int = 10, pretrained: bool = False, freeze_features: bool = False) -> nn.Module:
    """
    torchvision 내장 AlexNet을 CIFAR-10(10클래스)용으로 수정하여 반환.

    AlexNet 구조 요약:
    ┌──────────────────────────────────────────────────────────────┐
    │ features (Conv 블록 5개 + MaxPool)                           │
    │   Conv(3,64,11,s=4,p=2) → ReLU → MaxPool(3,2)              │
    │   Conv(64,192,5,p=2)    → ReLU → MaxPool(3,2)              │
    │   Conv(192,384,3,p=1)   → ReLU                             │
    │   Conv(384,256,3,p=1)   → ReLU                             │
    │   Conv(256,256,3,p=1)   → ReLU → MaxPool(3,2)              │
    ├──────────────────────────────────────────────────────────────┤
    │ avgpool: AdaptiveAvgPool2d(6×6)                              │
    ├──────────────────────────────────────────────────────────────┤
    │ classifier (FC 블록)                                         │
    │   Dropout → Linear(9216,4096) → ReLU                       │
    │   Dropout → Linear(4096,4096) → ReLU                       │
    │   Linear(4096, 1000)  ← 여기를 10으로 교체                  │
    └──────────────────────────────────────────────────────────────┘

    Args:
        num_classes:      출력 클래스 수 (CIFAR-10 = 10)
        pretrained:       True면 ImageNet 사전학습 가중치 로드
        freeze_features:  True면 features 파라미터를 고정 (전이학습 시 권장)

    Returns:
        수정된 AlexNet 모델
    """
    if pretrained:
        # ImageNet으로 사전학습된 가중치 로드 (torchvision 권장 방식)
        weights = AlexNet_Weights.DEFAULT
        model   = models.alexnet(weights=weights)
        print("✔ ImageNet 사전학습 가중치 로드 완료")
    else:
        # 가중치 없이 구조만 생성 (처음부터 학습)
        model = models.alexnet(weights=None)
        print("✔ 가중치 없이 AlexNet 구조 생성 (scratch)")

    # ── 마지막 분류 레이어 교체 (1000 → num_classes) ─────────────────────
    # AlexNet의 classifier[-1]은 Linear(4096, 1000) → 10클래스로 교체
    in_features = model.classifier[-1].in_features   # 4096
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    print(f"✔ classifier 마지막 FC: {in_features} → {num_classes} 클래스로 교체")

    # ── 전이학습: features 파라미터 고정 ────────────────────────────────
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False  # 컨볼루션 층 기울기 계산 비활성화
        print("✔ features 파라미터 고정 (classifier만 학습)")

    return model


# ══════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """테스트 세트 평균 손실과 정확도 반환."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits          = model(images)
            total_loss     += loss_fn(logits, labels).item() * images.size(0)
            total_correct  += (logits.argmax(dim=1) == labels).sum().item()
            total_count    += images.size(0)

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

        optimizer.zero_grad()              # 기울기 초기화
        logits = model(images)             # 순전파
        loss   = loss_fn(logits, labels)   # 손실 계산
        loss.backward()                    # 역전파
        optimizer.step()                   # 파라미터 업데이트

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count   += images.size(0)

    return total_loss / total_count, total_correct / total_count


# ══════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════
def plot_history(history: dict[str, list[float]], epochs: int, title: str, output_path: Path) -> None:
    """훈련 손실·정확도와 테스트 손실·정확도를 나란히 플롯."""
    x = range(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 손실 곡선
    ax1.plot(x, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(x, history["test_loss"],  label="Test Loss",  color="#2980b9", linewidth=2, linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 정확도 곡선
    ax2.plot(x, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#e74c3c", linewidth=2)
    ax2.plot(x, [a * 100 for a in history["test_acc"]],  label="Test Acc",  color="#2980b9", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy")
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
    parser = argparse.ArgumentParser(description="AlexNet on CIFAR-10 (torchvision).")
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=64)         # AlexNet은 메모리 사용량이 크므로 64 권장
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data-root",       type=Path,  default=Path("data"))
    parser.add_argument("--output-dir",      type=Path,  default=Path("outputs"))
    parser.add_argument("--pretrained",      action="store_true",            # 사전학습 가중치 사용 여부
                        help="ImageNet 사전학습 가중치 사용 (전이학습)")
    parser.add_argument("--freeze-features", action="store_true",            # features 파라미터 고정 여부
                        help="features 레이어 파라미터 고정 (--pretrained와 함께 사용)")
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"사용 장치: {device}")
    print(f"모드: {'사전학습(전이학습)' if args.pretrained else 'Scratch(처음부터 학습)'}\n")

    # ── 데이터 전처리 ─────────────────────────────────────────────────────
    # AlexNet 원본 입력 크기: 224×224
    # CIFAR-10 이미지(32×32)를 224×224로 업스케일
    if args.pretrained:
        # 사전학습 가중치 사용 시: ImageNet 표준 정규화 값 사용
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),   # ImageNet RGB 채널별 평균
            std =(0.229, 0.224, 0.225),   # ImageNet RGB 채널별 표준편차
        )
    else:
        # Scratch 학습 시: CIFAR-10 통계 사용
        normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616),
        )

    train_transform = transforms.Compose([
        transforms.Resize(224),                    # 32→224 업스케일 (AlexNet 입력 크기)
        transforms.RandomHorizontalFlip(),          # 데이터 증강: 좌우 반전
        transforms.RandomCrop(224, padding=8),     # 데이터 증강: 랜덤 크롭
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),                    # 평가 시에는 증강 없이 리사이즈만 적용
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(root=args.data_root, train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"훈련 샘플: {len(train_set):,}  /  테스트 샘플: {len(test_set):,}\n")

    # ── 모델 빌드 ─────────────────────────────────────────────────────────
    model   = build_alexnet(num_classes=10, pretrained=args.pretrained, freeze_features=args.freeze_features)
    model   = model.to(device)

    # 학습 가능한 파라미터 수 출력
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n전체 파라미터: {total:,}  /  학습 가능: {trainable:,}\n")

    # ── 옵티마이저·손실 함수 ─────────────────────────────────────────────
    # 학습 가능한(requires_grad=True) 파라미터만 옵티마이저에 전달
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # 스케줄러: 5 에폭마다 lr을 0.1배로 감소 (학습 안정화)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss_fn   = nn.CrossEntropyLoss()

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }

    print(f"{'Epoch':>6} {'LR':>8} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss,  test_acc  = evaluate(model, test_loader, loss_fn, device)

        scheduler.step()   # 에폭 종료 후 learning rate 업데이트

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"{epoch:>6d} {current_lr:>8.2e} {train_loss:>11.4f} "
            f"{train_acc*100:>9.2f}% {test_loss:>10.4f} {test_acc*100:>8.2f}%"
        )

    # ── 최종 결과 요약 ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("최종 결과 요약")
    print("=" * 62)
    print(f"  최고 테스트 정확도: {max(history['test_acc'])*100:.2f}%  (epoch {history['test_acc'].index(max(history['test_acc']))+1})")
    print(f"  최종 훈련 정확도:   {history['train_acc'][-1]*100:.2f}%")
    print(f"  최종 테스트 정확도: {history['test_acc'][-1]*100:.2f}%")

    # ── 그래프 저장 ───────────────────────────────────────────────────────
    mode_str = "Pretrained (Transfer Learning)" if args.pretrained else "Scratch"
    plot_history(
        history     = history,
        epochs      = args.epochs,
        title       = f"AlexNet on CIFAR-10 — {mode_str}",
        output_path = args.output_dir / with_script_prefix(__file__, "alexnet_cifar10_history.png"),
    )


if __name__ == "__main__":
    main()
