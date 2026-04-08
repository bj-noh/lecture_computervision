"""
01_01_sgd_adam_comparison.py
────────────────────────────────────────────────────────────────────────────
MNIST 데이터셋에서 SGD와 Adam 옵티마이저의 학습 성능을 비교합니다.

실행 예시:
    python 01_01_sgd_adam_comparison.py --epochs 10

출력:
    outputs/sgd_adam_comparison.png  ← 손실·정확도 비교 그래프
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
# 모델 정의 (01_mnist_mlp.py 와 동일한 구조)
# ══════════════════════════════════════════════════════════════════════════
class MnistMLP(nn.Module):
    """Flatten → Linear(784→512) → ReLU → Linear(512→10) 구조의 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),            # 28×28 → 784
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),      # 10개 클래스(0~9) 출력
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """테스트 세트 전체의 평균 손실과 정확도를 반환."""
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
) -> float:
    """한 에폭 학습 후 평균 훈련 손실을 반환."""
    model.train()
    total_loss, total_count = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # 기울기 초기화
        logits = model(images)              # 순전파
        loss   = loss_fn(logits, labels)    # 손실 계산
        loss.backward()                     # 역전파
        optimizer.step()                    # 파라미터 업데이트

        total_loss  += loss.item() * images.size(0)
        total_count += images.size(0)

    return total_loss / total_count


def run_experiment(
    optimizer_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,     # SGD 전용 모멘텀 계수
) -> dict[str, list[float]]:
    """
    지정한 옵티마이저로 MnistMLP를 학습하고 에폭별 기록을 반환.

    Returns:
        {
          'train_loss': [에폭별 훈련 손실],
          'test_loss':  [에폭별 테스트 손실],
          'test_acc':   [에폭별 테스트 정확도],
        }
    """
    # ── 모델·손실함수 초기화 ──────────────────────────────────────────────
    model   = MnistMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # ── 옵티마이저 선택 ───────────────────────────────────────────────────
    if optimizer_name.upper() == "SGD":
        # 모멘텀 SGD: 기울기 방향에 관성을 추가해 수렴 안정화
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name.upper() == "ADAM":
        # Adam: 1차·2차 모멘트를 적응적으로 조정하는 옵티마이저
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss              = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc     = evaluate(model, test_loader, device, loss_fn)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"[{optimizer_name:4s}] epoch={epoch:02d} "
            f"train_loss={train_loss:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"test_acc={test_acc:.4f}"
        )

    return history


# ══════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════
def plot_comparison(
    results: dict[str, dict[str, list[float]]],
    epochs: int,
    output_path: Path,
) -> None:
    """
    SGD·Adam의 훈련 손실 / 테스트 손실 / 테스트 정확도를 한 장에 비교.

    Args:
        results:     {'SGD': history_dict, 'Adam': history_dict}
        epochs:      총 에폭 수 (x축 범위)
        output_path: 그래프 저장 경로
    """
    x = range(1, epochs + 1)

    # 색상·스타일 정의
    styles = {
        "SGD":  {"color": "#e74c3c", "linestyle": "-",  "marker": "o"},
        "Adam": {"color": "#2980b9", "linestyle": "--", "marker": "s"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("SGD vs Adam on MNIST (MLP)", fontsize=14, fontweight="bold")

    titles    = ["Train Loss", "Test Loss", "Test Accuracy"]
    keys      = ["train_loss", "test_loss", "test_acc"]
    ylabels   = ["Loss", "Loss", "Accuracy"]

    for ax, title, key, ylabel in zip(axes, titles, keys, ylabels):
        for opt_name, history in results.items():
            s = styles[opt_name]
            ax.plot(
                x, history[key],
                label=opt_name,
                color=s["color"],
                linestyle=s["linestyle"],
                marker=s["marker"],
                markersize=4,
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(x))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\n그래프 저장 완료: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── 인자 파싱 ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Compare SGD vs Adam on MNIST.")
    parser.add_argument("--epochs",     type=int,   default=10)          # 학습 에폭 수
    parser.add_argument("--batch-size", type=int,   default=128)         # 미니배치 크기
    parser.add_argument("--lr-sgd",     type=float, default=1e-2)        # SGD 학습률 (Adam보다 크게 설정)
    parser.add_argument("--lr-adam",    type=float, default=1e-3)        # Adam 학습률
    parser.add_argument("--momentum",   type=float, default=0.9)         # SGD 모멘텀
    parser.add_argument("--data-root",  type=Path,  default=Path("data"))
    parser.add_argument("--output-dir", type=Path,  default=Path("outputs"))
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"사용 장치: {device}\n")

    # ── 데이터 준비 ───────────────────────────────────────────────────────
    # MNIST 공식 통계값으로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root=args.data_root, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)

    # ── 실험: SGD ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"[SGD]  lr={args.lr_sgd}, momentum={args.momentum}")
    print("=" * 60)
    sgd_history = run_experiment(
        optimizer_name="SGD",
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr_sgd,
        momentum=args.momentum,
    )

    # ── 실험: Adam ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[Adam] lr={args.lr_adam}")
    print("=" * 60)
    adam_history = run_experiment(
        optimizer_name="Adam",
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr_adam,
        momentum=0.0,    # Adam은 모멘텀 파라미터 불필요
    )

    # ── 최종 결과 요약 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 테스트 결과 요약")
    print("=" * 60)
    print(f"  SGD  → test_loss={sgd_history['test_loss'][-1]:.4f},  test_acc={sgd_history['test_acc'][-1]:.4f}")
    print(f"  Adam → test_loss={adam_history['test_loss'][-1]:.4f},  test_acc={adam_history['test_acc'][-1]:.4f}")

    # ── 비교 그래프 저장 ──────────────────────────────────────────────────
    plot_comparison(
        results={"SGD": sgd_history, "Adam": adam_history},
        epochs=args.epochs,
        output_path=args.output_dir / with_script_prefix(__file__, "sgd_adam_comparison.png"),
    )


if __name__ == "__main__":
    main()
