"""
02_06_densenet_dogs.py
────────────────────────────────────────────────────────────────────────────
Pretrained DenseNet으로 견종(개 품종)을 인식합니다.

DenseNet (Huang et al., 2017) 핵심 아이디어 — Dense Connection:
  ResNet:   x_l = F_l(x_{l-1}) + x_{l-1}    (직전 레이어만 연결)
  DenseNet: x_l = F_l([x_0, x_1, ..., x_{l-1}])  (모든 이전 레이어 연결)

  장점:
    - 기울기 소실 문제 완화 (shortcut이 모든 레이어에 연결)
    - 피처 재사용(feature reuse)으로 파라미터 효율↑
    - 학습 초기부터 풍부한 피처 활용 가능

DenseNet 계열:
  densenet121: Growth rate=32, [6,12,24,16] Dense Block
  densenet161: Growth rate=48, [6,12,36,24] Dense Block
  densenet169: Growth rate=32, [6,12,32,32] Dense Block
  densenet201: Growth rate=32, [6,12,48,32] Dense Block

데이터셋: Oxford-IIIT Pet (개 품종 25종만 필터링)
  - 원본: 37종 반려동물 (개 25종 + 고양이 12종)
  - 견종만 추출: 25개 클래스, 클래스당 약 200장
  - 이미지 크기: 다양 → 224×224 리사이즈
  - 클래스당 샘플 수가 적어 사전학습(ImageNet)의 효과를 잘 확인 가능

실행 예시:
    python 02_06_densenet_dogs.py --arch densenet121 --epochs 20
    python 02_06_densenet_dogs.py --arch densenet201 --epochs 15 --freeze-features

출력:
    outputs/densenet_dogs_{arch}_history.png
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import (
    DenseNet121_Weights, DenseNet161_Weights,
    DenseNet169_Weights, DenseNet201_Weights,
)

from output_naming import with_script_prefix


# ══════════════════════════════════════════════════════════════════════════
# Oxford-IIIT Pet 데이터셋에서 개 품종 25종 클래스명
# ══════════════════════════════════════════════════════════════════════════
# Oxford-IIIT Pet의 클래스 이름 규칙: 대문자로 시작하면 고양이, 소문자로 시작하면 개
# (예: 'Abyssinian' → 고양이, 'american_bulldog' → 개)
# torchvision이 반환하는 클래스 목록에서 소문자 시작 = 견종 필터링

def get_dog_indices(dataset: datasets.OxfordIIITPet) -> tuple[list[int], list[str], dict[int, int]]:
    """
    OxfordIIITPet 데이터셋에서 개 품종 샘플의 인덱스를 추출.

    Oxford-IIIT Pet 클래스 규칙:
      - 소문자로 시작하는 클래스 → 개 품종 (예: 'beagle', 'pug')
      - 대문자로 시작하는 클래스 → 고양이 품종 (예: 'Abyssinian', 'Bengal')

    Returns:
        sample_indices: 개 품종 샘플의 데이터셋 내 인덱스 리스트
        dog_classes:    개 품종 클래스 이름 리스트 (알파벳순)
        label_remap:    원본 레이블 → 0부터 시작하는 새 레이블 매핑
    """
    all_classes = dataset.classes  # 전체 37 클래스 이름 목록

    # Oxford-IIIT Pet의 binary-category:
    #   0 = cat, 1 = dog
    dog_orig_labels = sorted({
        label for label, species in zip(dataset._labels, dataset._bin_labels)
        if species == 1
    })
    dog_classes = [all_classes[label] for label in dog_orig_labels]

    # 원본 레이블 → 새 레이블 (0~24) 매핑
    label_remap = {old: new for new, old in enumerate(dog_orig_labels)}

    # 개 품종 샘플 인덱스 추출
    sample_indices = [
        i for i, species in enumerate(dataset._bin_labels)
        if species == 1
    ]

    return sample_indices, dog_classes, label_remap


class DogSubset(torch.utils.data.Dataset):
    """
    OxfordIIITPet에서 개 품종만 추출한 서브셋 Dataset.
    레이블을 0부터의 연속 정수로 재매핑.
    """

    def __init__(self, base_dataset: datasets.OxfordIIITPet,
                 indices: list[int], label_remap: dict[int, int]) -> None:
        self.base      = base_dataset
        self.indices   = indices
        self.label_remap = label_remap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = self.base[self.indices[idx]]
        label = target[0] if isinstance(target, tuple) else target
        return image, self.label_remap[label]    # 레이블 재매핑


# ══════════════════════════════════════════════════════════════════════════
# 지원 아키텍처 설정
# ══════════════════════════════════════════════════════════════════════════
ARCH_CONFIG: dict[str, tuple] = {
    "densenet121": (models.densenet121, DenseNet121_Weights.DEFAULT),
    "densenet161": (models.densenet161, DenseNet161_Weights.DEFAULT),
    "densenet169": (models.densenet169, DenseNet169_Weights.DEFAULT),
    "densenet201": (models.densenet201, DenseNet201_Weights.DEFAULT),
}

ARCH_INFO: dict[str, str] = {
    "densenet121": "Growth=32, Blocks=[6,12,24,16], 파라미터 ~8M",
    "densenet161": "Growth=48, Blocks=[6,12,36,24], 파라미터 ~29M",
    "densenet169": "Growth=32, Blocks=[6,12,32,32], 파라미터 ~14M",
    "densenet201": "Growth=32, Blocks=[6,12,48,32], 파라미터 ~20M",
}


# ══════════════════════════════════════════════════════════════════════════
# 모델 빌드
# ══════════════════════════════════════════════════════════════════════════
def build_densenet(
    arch: str,
    num_classes: int,
    freeze_features: bool = False,
) -> nn.Module:
    """
    ImageNet 사전학습 DenseNet을 견종 분류용으로 수정하여 반환.

    DenseNet의 분류기 구조 (torchvision 기본):
      model.classifier = Linear(in_features, 1000)
        - densenet121/169: in_features = 1024
        - densenet161:     in_features = 2208
        - densenet201:     in_features = 1920
      → Linear(in_features, num_classes) 로 교체

    Args:
        arch:            DenseNet 변형
        num_classes:     출력 클래스 수 (개 품종 수)
        freeze_features: True면 classifier 이전 레이어 파라미터 고정
    """
    builder, weights = ARCH_CONFIG[arch]

    # ── 항상 사전학습 가중치 사용 (Pretrained DenseNet) ──────────────────
    model = builder(weights=weights)
    print(f"✔ {arch.upper()} ImageNet 사전학습 가중치 로드 완료")
    print(f"  구성: {ARCH_INFO[arch]}")

    # ── 분류기 교체: 1000 → num_classes ─────────────────────────────────
    # DenseNet은 VGG/ResNet과 달리 분류기가 단순한 Linear 하나
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    print(f"✔ classifier 교체: Linear({in_features}, 1000) → Linear({in_features}, {num_classes})")

    # ── 전이학습: features 파라미터 고정 ────────────────────────────────
    if freeze_features:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False   # Dense Block 기울기 계산 비활성화
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


def plot_sample_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    dog_classes: list[str],
    device: torch.device,
    output_path: Path,
    n: int = 9,
) -> None:
    """
    테스트 이미지 n장에 대한 예측 결과를 시각화하여 저장.
    맞춘 경우 초록색, 틀린 경우 빨간색 제목으로 표시.
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images_dev = images[:n].to(device)

    with torch.no_grad():
        preds = model(images_dev).argmax(dim=1).cpu()

    # ImageNet 정규화 역변환 (시각화용)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    fig.suptitle("Pretrained DenseNet — Dog Breed Predictions", fontsize=13, fontweight="bold")

    for ax, img, label, pred in zip(axes.flat, images[:n], labels[:n], preds):
        img_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img_show)
        color = "green" if label.item() == pred.item() else "red"
        ax.set_title(
            f"GT:   {dog_classes[label.item()]}\n"
            f"Pred: {dog_classes[pred.item()]}",
            color=color, fontsize=8,
        )
        ax.axis("off")

    # 남는 서브플롯 숨김
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"샘플 예측 이미지 저장 완료: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── 인자 파싱 ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Pretrained DenseNet for Dog Breed Recognition (Oxford-IIIT Pet)."
    )
    parser.add_argument("--arch",            type=str,   default="densenet121",
                        choices=list(ARCH_CONFIG.keys()),
                        help="DenseNet 변형 (densenet121 | densenet161 | densenet169 | densenet201)")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--data-root",       type=Path,  default=Path("data"))
    parser.add_argument("--output-dir",      type=Path,  default=Path("outputs"))
    parser.add_argument("--freeze-features", action="store_true",
                        help="classifier 이전 레이어 파라미터 고정 (빠른 전이학습)")
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"사용 장치: {device}")
    print(f"아키텍처:  {args.arch.upper()} (Pretrained on ImageNet)")
    print(f"데이터셋:  Oxford-IIIT Pet — 개 품종 25종 필터링\n")

    # ── 데이터 전처리 ─────────────────────────────────────────────────────
    # 사전학습 DenseNet → ImageNet 정규화 통계 사용 필수
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),    # 랜덤 크롭+리사이즈 (강력한 증강)
        transforms.RandomHorizontalFlip(),     # 좌우 반전
        transforms.ColorJitter(               # 색상 변형 (반려동물 털 색상 다양성 대응)
            brightness=0.3, contrast=0.2, saturation=0.2
        ),
        transforms.RandomRotation(15),        # ±15도 랜덤 회전
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # ── 데이터셋 로드 및 개 품종 필터링 ──────────────────────────────────
    print("Oxford-IIIT Pet 데이터셋 로드 중...")
    raw_train = datasets.OxfordIIITPet(
        root=args.data_root,
        split="trainval",
        target_types=["category", "binary-category"],
        download=True,
        transform=train_transform,
    )
    raw_test  = datasets.OxfordIIITPet(
        root=args.data_root,
        split="test",
        target_types=["category", "binary-category"],
        download=True,
        transform=test_transform,
    )

    # 개 품종 인덱스와 레이블 매핑 추출
    train_idx, dog_classes, label_remap = get_dog_indices(raw_train)
    test_idx,  _,           _           = get_dog_indices(raw_test)

    # 개 품종만 포함하는 서브셋 생성
    train_set = DogSubset(raw_train, train_idx, label_remap)
    test_set  = DogSubset(raw_test,  test_idx,  label_remap)

    num_dog_classes = len(dog_classes)
    print(f"\n개 품종 클래스 수: {num_dog_classes}종")
    print(f"품종 목록: {', '.join(dog_classes)}")
    print(f"\n훈련 샘플: {len(train_set):,}  /  테스트 샘플: {len(test_set):,}\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── 모델 빌드 ─────────────────────────────────────────────────────────
    model = build_densenet(
        arch            = args.arch,
        num_classes     = num_dog_classes,
        freeze_features = args.freeze_features,
    )
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n전체 파라미터: {total:,}  /  학습 가능: {trainable:,}\n")

    # ── 옵티마이저·스케줄러·손실 함수 ────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    # ReduceLROnPlateau: 검증 손실이 개선되지 않으면 lr 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)   # Label Smoothing: 소수 데이터 과적합 억제

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

        # ReduceLROnPlateau는 모니터링 값을 직접 전달
        scheduler.step(test_acc)

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
    print(f"최종 결과 요약 ({args.arch.upper()} 견종 인식)")
    print("=" * 62)
    print(f"  개 품종 수:         {num_dog_classes}종")
    print(f"  최고 테스트 정확도: {best_test_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  최종 훈련 정확도:   {history['train_acc'][-1]*100:.2f}%")
    print(f"  최종 테스트 정확도: {history['test_acc'][-1]*100:.2f}%")

    # ── 그래프 및 샘플 예측 저장 ─────────────────────────────────────────
    plot_history(
        history     = history,
        epochs      = args.epochs,
        title       = f"{args.arch.upper()} 견종 인식 — Pretrained (Oxford-IIIT Pet Dog Breeds)",
        output_path = args.output_dir / with_script_prefix(__file__, f"densenet_dogs_{args.arch}_history.png"),
    )

    plot_sample_predictions(
        model       = model,
        test_loader = test_loader,
        dog_classes = dog_classes,
        device      = device,
        output_path = args.output_dir / with_script_prefix(__file__, f"densenet_dogs_{args.arch}_predictions.png"),
        n           = 9,
    )


if __name__ == "__main__":
    main()
