"""
02_04_vggnet_cifar100.py
────────────────────────────────────────────────────────────────────────────
torchvision.models 의 VGGNet 계열(VGG-11/13/16/19)을 사용해
CIFAR-100 데이터셋을 분류합니다.

VGGNet (Simonyan & Zisserman, 2014) 핵심 설계 철학:
  - 모든 Conv 레이어를 3×3 (stride=1, padding=1) 으로 통일
  - 깊이(depth)를 늘려 표현력 향상
  - 작은 필터를 여러 겹 쌓으면 큰 필터와 수용 영역이 같지만 파라미터 수는 더 적음
    예: 3×3 두 번 = 5×5와 동일 수용 영역, 파라미터는 18 vs 25

VGG 계열 변형:
  vgg11: [1,1,2,2,2] Conv 블록 → 11층
  vgg13: [2,2,2,2,2] Conv 블록 → 13층
  vgg16: [2,2,3,3,3] Conv 블록 → 16층 ← 가장 널리 사용
  vgg19: [2,2,4,4,4] Conv 블록 → 19층

CIFAR-100 데이터셋:
  - 100개 클래스, 클래스당 500장 훈련 / 100장 테스트
  - 총 50,000장 훈련 / 10,000장 테스트
  - 이미지 크기: 32×32×3 → 224×224 리사이즈

실행 예시:
    python 02_04_vggnet_cifar100.py --arch vgg16 --epochs 20
    python 02_04_vggnet_cifar100.py --arch vgg11 --epochs 20 --pretrained

출력:
    outputs/vggnet_cifar100_{arch}_{mode}_history.png
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
    VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights,
)

from output_naming import with_script_prefix


# ══════════════════════════════════════════════════════════════════════════
# 지원 아키텍처 설정
# ══════════════════════════════════════════════════════════════════════════
# VGG 변형별 모델 빌더와 사전학습 가중치를 한 곳에서 관리
ARCH_CONFIG: dict[str, tuple] = {
    "vgg11": (models.vgg11, VGG11_Weights.DEFAULT),
    "vgg13": (models.vgg13, VGG13_Weights.DEFAULT),
    "vgg16": (models.vgg16, VGG16_Weights.DEFAULT),
    "vgg19": (models.vgg19, VGG19_Weights.DEFAULT),
}


# ══════════════════════════════════════════════════════════════════════════
# 모델 빌드
# ══════════════════════════════════════════════════════════════════════════
def build_vgg(
    arch: str,
    num_classes: int = 100,
    pretrained: bool = False,
    freeze_features: bool = False,
) -> nn.Module:
    """
    torchvision 내장 VGG 모델을 CIFAR-100 (100클래스)용으로 수정하여 반환.

    VGGNet의 classifier 구조 (torchvision 기본):
        Linear(25088, 4096) → ReLU → Dropout
        Linear(4096,  4096) → ReLU → Dropout
        Linear(4096,  1000)          ← 이 레이어를 num_classes로 교체

    Args:
        arch:            VGG 변형 ('vgg11' | 'vgg13' | 'vgg16' | 'vgg19')
        num_classes:     출력 클래스 수 (CIFAR-100 = 100)
        pretrained:      ImageNet 사전학습 가중치 사용 여부
        freeze_features: True면 features 레이어 파라미터 고정
    """
    if arch not in ARCH_CONFIG:
        raise ValueError(f"지원하지 않는 아키텍처: {arch}. {list(ARCH_CONFIG.keys())} 중 선택하세요.")

    builder, weights = ARCH_CONFIG[arch]

    if pretrained:
        model = builder(weights=weights)
        print(f"✔ {arch.upper()} ImageNet 사전학습 가중치 로드 완료")
    else:
        model = builder(weights=None)
        print(f"✔ {arch.upper()} 가중치 없이 생성 (scratch)")

    # ── 마지막 분류 레이어 교체: 1000 → num_classes ─────────────────────
    # VGGNet의 classifier[-1]은 Linear(4096, 1000)
    in_features = model.classifier[-1].in_features   # 4096
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    print(f"✔ classifier 마지막 FC: {in_features} → {num_classes} 클래스로 교체")

    # ── 전이학습: features 고정 ──────────────────────────────────────────
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False  # 컨볼루션 레이어 기울기 계산 비활성화
        print("✔ features 파라미터 고정 (classifier만 학습)")

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
    """테스트 세트 평균 손실과 Top-1 정확도를 반환."""
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


def output_tag(pretrained: bool, freeze_features: bool) -> str:
    """실험 설정별 출력 파일 구분 태그를 반환."""
    if not pretrained:
        return "scratch"
    if freeze_features:
        return "pretrained_frozen"
    return "pretrained"


def mode_labels(pretrained: bool, freeze_features: bool) -> tuple[str, str]:
    """콘솔 출력용/플롯용 실행 모드 라벨을 반환."""
    if not pretrained:
        return "Scratch", "Scratch"
    if freeze_features:
        return "전이학습 (Pretrained, Features Frozen)", "Pretrained (Frozen Features)"
    return "전이학습 (Pretrained)", "Pretrained"


# ══════════════════════════════════════════════════════════════════════════
# 아키텍처 정보 출력
# ══════════════════════════════════════════════════════════════════════════
def print_arch_summary(model: nn.Module, arch: str) -> None:
    """VGG 변형별 레이어 구성과 파라미터 수를 요약 출력."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Conv 레이어 수 카운트
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))

    print(f"\n{'='*50}")
    print(f"아키텍처: {arch.upper()}")
    print(f"  Conv 레이어 수: {n_conv}")
    print(f"  전체 파라미터:  {total:,}")
    print(f"  학습 가능:      {trainable:,}")
    print(f"{'='*50}\n")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── 인자 파싱 ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="VGGNet on CIFAR-100 (torchvision).")
    parser.add_argument("--arch",            type=str,   default="vgg16",
                        choices=list(ARCH_CONFIG.keys()),
                        help="VGG 변형 선택 (vgg11 | vgg13 | vgg16 | vgg19)")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data-root",       type=Path,  default=Path("data"))
    parser.add_argument("--output-dir",      type=Path,  default=Path("outputs"))
    parser.add_argument("--pretrained",      action="store_true",
                        help="ImageNet 사전학습 가중치 사용 (전이학습)")
    parser.add_argument("--freeze-features", action="store_true",
                        help="features 파라미터 고정 (--pretrained와 함께 사용)")
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_mode, plot_mode = mode_labels(args.pretrained, args.freeze_features)
    print(f"사용 장치: {device}")
    print(f"아키텍처:  {args.arch.upper()}")
    print(f"모드:      {display_mode}")
    print(f"데이터셋:  CIFAR-100 (100 classes, 32×32 → 224×224)\n")

    # ── 데이터 전처리 ─────────────────────────────────────────────────────
    # VGGNet 원본 입력 크기: 224×224
    # CIFAR-100 이미지(32×32)를 224×224로 업스케일

    if args.pretrained:
        # ImageNet 정규화 통계 사용 (사전학습 가중치와 동일한 분포 유지)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std =(0.229, 0.224, 0.225),
        )
    else:
        # CIFAR-100 통계 사용
        normalize = transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std =(0.2675, 0.2565, 0.2761),
        )

    train_transform = transforms.Compose([
        transforms.Resize(224),                  # 32 → 224
        transforms.RandomHorizontalFlip(),        # 좌우 반전 증강
        transforms.RandomCrop(224, padding=16),  # 랜덤 크롭 증강
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR100(root=args.data_root, train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"훈련 샘플: {len(train_set):,}  /  테스트 샘플: {len(test_set):,}")

    # ── 모델 빌드 및 정보 출력 ────────────────────────────────────────────
    model = build_vgg(
        arch            = args.arch,
        num_classes     = 100,
        pretrained      = args.pretrained,
        freeze_features = args.freeze_features,
    )
    model = model.to(device)
    print_arch_summary(model, args.arch)

    # ── 옵티마이저·스케줄러·손실 함수 ────────────────────────────────────
    # 학습 가능한 파라미터만 옵티마이저에 전달 (freeze 시 효율적)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,   # L2 정규화
    )
    # MultiStepLR: 지정 에폭에서 lr을 0.1배로 감소
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    loss_fn    = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing으로 과적합 억제

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
    print(f"최종 결과 요약 ({args.arch.upper()} on CIFAR-100)")
    print("=" * 62)
    print(f"  최고 테스트 정확도: {best_test_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  최종 훈련 정확도:   {history['train_acc'][-1]*100:.2f}%")
    print(f"  최종 테스트 정확도: {history['test_acc'][-1]*100:.2f}%")

    # ── 그래프 저장 ───────────────────────────────────────────────────────
    run_tag = output_tag(args.pretrained, args.freeze_features)
    plot_history(
        history     = history,
        epochs      = args.epochs,
        title       = f"{args.arch.upper()} on CIFAR-100 - {plot_mode}",
        output_path = args.output_dir / with_script_prefix(
            __file__,
            f"vggnet_cifar100_{args.arch}_{run_tag}_history.png",
        ),
    )


if __name__ == "__main__":
    main()
