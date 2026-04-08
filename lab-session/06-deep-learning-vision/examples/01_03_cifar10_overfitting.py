"""
01_02_cifar10_overfitting.py
────────────────────────────────────────────────────────────────────────────
깊은 다층 퍼셉트론(Deep MLP)으로 CIFAR-10을 학습해 오버피팅을 시각화합니다.

참고: 교재 [프로그램 7-6] (TensorFlow/Keras) → PyTorch 변환 버전

모델 구조 (교재와 동일):
    Flatten → Linear(3072→1024) → ReLU
            → Linear(1024→512) → ReLU
            → Linear(512→512)  → ReLU
            → Linear(512→10)

오버피팅이 나타나는 이유:
    - MLP는 이미지의 공간적 구조(spatial structure)를 무시
    - 파라미터 수가 CIFAR-10의 복잡도에 비해 매우 많음
    - 정규화(Dropout, BatchNorm 등)를 전혀 사용하지 않음
    → 훈련 손실은 계속 내려가지만 검증 손실은 어느 순간부터 증가

실행 예시:
    python 01_02_cifar10_overfitting.py --epochs 50

출력:
    outputs/cifar10_overfitting.png  ← 오버피팅 곡선 그래프
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from output_naming import with_script_prefix


# ══════════════════════════════════════════════════════════════════════════
# 모델 정의  (교재 프로그램 7-6과 동일한 구조)
# ══════════════════════════════════════════════════════════════════════════
class DeepMLP(nn.Module):
    """
    CIFAR-10용 깊은 MLP (Deep Multilayer Perceptron).

    교재 TensorFlow 코드를 PyTorch로 그대로 옮긴 버전.
    정규화 없음 → 오버피팅 의도적으로 유발.

    입력: 32×32×3 컬러 이미지 → 3072차원 벡터
    출력: 10개 클래스 로짓
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),             # 32×32×3 → 3072

            # ── 교재 18~21행과 동일한 층 구성 ──────────────────────────
            nn.Linear(3072, 1024),   # 첫 번째 은닉층: 3072 → 1024
            nn.ReLU(),
            nn.Linear(1024, 512),    # 두 번째 은닉층: 1024 → 512
            nn.ReLU(),
            nn.Linear(512, 512),     # 세 번째 은닉층: 512 → 512
            nn.ReLU(),
            nn.Linear(512, 10),      # 출력층: 512 → 10 클래스
            # ※ nn.CrossEntropyLoss가 내부적으로 Softmax 포함하므로 생략
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """검증/테스트 세트의 평균 손실과 정확도를 반환."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
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
def plot_overfitting(history: dict[str, list[float]], epochs: int, output_path: Path) -> None:
    """
    훈련 vs 검증 손실·정확도 곡선을 나란히 그려 오버피팅을 시각화.

    오버피팅 신호:
      - 훈련 손실(train loss)은 계속 감소
      - 검증 손실(test loss)은 특정 에폭 이후 증가
      - 두 곡선 사이의 간격이 벌어질수록 심한 오버피팅
    """
    x = range(1, epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Deep MLP on CIFAR-10 — Overfitting Demonstration\n"
        "(No Regularization: No Dropout · No BatchNorm · No Weight Decay)",
        fontsize=12, fontweight="bold",
    )

    # ── 손실 곡선 ─────────────────────────────────────────────────────────
    ax1.plot(x, history["train_loss"], label="Train Loss",
             color="#e74c3c", linewidth=2)
    ax1.plot(x, history["test_loss"],  label="Test Loss",
             color="#2980b9", linewidth=2, linestyle="--")

    # 오버피팅 시작 지점 강조 (test_loss 최솟값 에폭)
    best_epoch = history["test_loss"].index(min(history["test_loss"])) + 1
    ax1.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
                label=f"Best Test Loss (epoch {best_epoch})")
    ax1.fill_between(
        range(best_epoch, epochs + 1),
        history["train_loss"][best_epoch - 1:],
        history["test_loss"][best_epoch - 1:],
        alpha=0.15, color="red", label="Overfitting Gap",
    )
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 정확도 곡선 ───────────────────────────────────────────────────────
    ax2.plot(x, [a * 100 for a in history["train_acc"]], label="Train Acc",
             color="#e74c3c", linewidth=2)
    ax2.plot(x, [a * 100 for a in history["test_acc"]],  label="Test Acc",
             color="#2980b9", linewidth=2, linestyle="--")
    ax2.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
                label=f"Best Test Loss (epoch {best_epoch})")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\n그래프 저장 완료: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── 인자 파싱 ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Deep MLP on CIFAR-10: overfitting demonstration."
    )
    parser.add_argument("--epochs",     type=int,   default=50)        # 오버피팅을 보려면 에폭 수를 충분히 크게
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)      # Adam 학습률
    parser.add_argument("--data-root",  type=Path,  default=Path("data"))
    parser.add_argument("--output-dir", type=Path,  default=Path("outputs"))
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"사용 장치: {device}")

    # ── 데이터 전처리 ─────────────────────────────────────────────────────
    # 교재: x/255.0 (단순 스케일링) → ToTensor()가 동일하게 [0,1]로 변환
    # CIFAR-10 공식 통계로 추가 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),                                   # [0,255] → [0,1]
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),                      # CIFAR-10 채널별 평균
            std =(0.2470, 0.2435, 0.2616),                      # CIFAR-10 채널별 표준편차
        ),
    ])

    train_set = datasets.CIFAR10(root=args.data_root, train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"훈련 샘플: {len(train_set):,}  /  테스트 샘플: {len(test_set):,}\n")

    # ── 모델·옵티마이저·손실 함수 초기화 ─────────────────────────────────
    model     = DeepMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # ※ weight_decay=0 으로 L2 정규화 없음 → 오버피팅 의도적으로 유발
    loss_fn   = nn.CrossEntropyLoss()

    # 모델 파라미터 수 출력
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 총 파라미터 수: {n_params:,}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9}")
    print("-" * 55)

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss,  test_acc  = evaluate(model, test_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # 오버피팅 경고 표시 (train_acc > test_acc + 10%)
        gap_flag = " ⚠ OVERFIT" if (train_acc - test_acc) > 0.10 else ""
        print(
            f"{epoch:>6d} {train_loss:>11.4f} {train_acc*100:>9.2f}% "
            f"{test_loss:>10.4f} {test_acc*100:>8.2f}%{gap_flag}"
        )

    # ── 최종 요약 ─────────────────────────────────────────────────────────
    best_epoch     = history["test_loss"].index(min(history["test_loss"])) + 1
    final_gap_acc  = (history["train_acc"][-1] - history["test_acc"][-1]) * 100
    final_gap_loss = history["test_loss"][-1] - history["train_loss"][-1]

    print("\n" + "=" * 55)
    print("최종 결과 요약")
    print("=" * 55)
    print(f"  최적 테스트 손실 에폭:  epoch {best_epoch}")
    print(f"  최종 훈련  정확도:      {history['train_acc'][-1]*100:.2f}%")
    print(f"  최종 테스트 정확도:     {history['test_acc'][-1]*100:.2f}%")
    print(f"  정확도 갭 (train-test): {final_gap_acc:.2f}%p  ← 오버피팅 정도")
    print(f"  손실 갭   (test-train): {final_gap_loss:.4f}    ← 오버피팅 정도")

    # ── 그래프 저장 ───────────────────────────────────────────────────────
    plot_overfitting(
        history=history,
        epochs=args.epochs,
        output_path=args.output_dir / with_script_prefix(__file__, "cifar10_overfitting.png"),
    )


if __name__ == "__main__":
    main()
