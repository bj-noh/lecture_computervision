from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from output_naming import with_script_prefix


class MnistMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 네트워크 구조 정의: Flatten → Linear(784→512) → ReLU → Linear(512→10)
        # 입력: 28×28 흑백 이미지 (784차원)
        # 출력: 10개 클래스(0~9)에 대한 로짓(logit) 값
        self.net = nn.Sequential(
            nn.Flatten(),             # 2D 이미지를 1D 벡터로 펼침 (28×28 → 784)
            nn.Linear(28 * 28, 512), # 첫 번째 완전연결층: 784 → 512
            nn.ReLU(),               # 비선형 활성화 함수
            nn.Linear(512, 10),      # 두 번째 완전연결층: 512 → 10 (클래스 수)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파(forward pass): 입력 이미지를 받아 클래스별 로짓을 반환."""
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    모델 성능 평가 함수.

    Args:
        model:  평가할 PyTorch 모델
        loader: 평가용 DataLoader
        device: 연산 장치 (CPU 또는 GPU)

    Returns:
        (평균 손실, 정확도) 튜플
    """
    model.eval()  # 평가 모드: Dropout·BatchNorm 등 비활성화
    loss_fn = nn.CrossEntropyLoss()  # 다중 클래스 분류용 손실 함수
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():  # 역전파 불필요 → 메모리·속도 최적화
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)                              # 예측 로짓
            # 배치 크기 가중 손실 누적 (나중에 평균 계산을 위해)
            total_loss += loss_fn(logits, labels).item() * images.size(0)
            # 예측 클래스(argmax)와 실제 레이블 비교하여 정답 수 누적
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

    # 전체 샘플에 대한 평균 손실과 정확도 반환
    return total_loss / total_count, total_correct / total_count


def save_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    """
    테스트 이미지 9장에 대한 예측 결과를 3×3 그리드 이미지로 저장.

    Args:
        model:       예측에 사용할 모델
        loader:      테스트 DataLoader
        device:      연산 장치
        output_path: 결과 이미지 저장 경로 (.png)
    """
    model.eval()
    # DataLoader에서 첫 번째 배치를 가져와 앞 9장만 사용
    images, labels = next(iter(loader))
    images = images[:9].to(device)
    labels = labels[:9]

    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()  # 각 이미지의 예측 클래스

    # 3×3 서브플롯 생성
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, image, label, pred in zip(axes.flat, images.cpu(), labels, preds):
        ax.imshow(image.squeeze(0), cmap="gray")           # 채널 차원 제거 후 흑백 표시
        ax.set_title(f"gt={label.item()} pred={pred.item()}")  # 정답(gt)과 예측(pred) 표시
        ax.axis("off")                                      # 축 눈금 제거

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)  # 고해상도로 저장
    plt.close(fig)                     # 메모리 해제


def main() -> None:
    """학습 파이프라인 진입점: 인자 파싱 → 데이터 로딩 → 학습 → 평가 → 결과 저장."""

    # ── 커맨드라인 인자 파싱 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST.")
    parser.add_argument("--epochs",     type=int,   default=5)          # 학습 에폭 수
    parser.add_argument("--batch-size", type=int,   default=128)        # 미니배치 크기
    parser.add_argument("--lr",         type=float, default=1e-3)       # 학습률 (Adam)
    parser.add_argument("--data-root",  type=Path,  default=Path("data"))     # 데이터 저장 경로
    parser.add_argument("--output-dir", type=Path,  default=Path("outputs"))  # 결과 저장 디렉터리
    args = parser.parse_args()

    # ── 장치 설정 및 출력 디렉터리 생성 ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 전처리 및 로딩 ────────────────────────────────────────────
    # MNIST 공식 통계(평균=0.1307, 표준편차=0.3081)로 픽셀값 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),                          # PIL 이미지 → [0,1] 텐서
        transforms.Normalize((0.1307,), (0.3081,)),    # 표준화 (z-score)
    ])
    train_set = datasets.MNIST(root=args.data_root, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)   # 학습: 셔플 O
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)  # 평가: 셔플 X

    # ── 모델·옵티마이저·손실 함수 초기화 ─────────────────────────────────
    model     = MnistMLP().to(device)                            # 모델을 지정 장치로 이동
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # Adam 옵티마이저
    loss_fn   = nn.CrossEntropyLoss()                            # 소프트맥스 + NLL 손실

    # ── 학습 루프 ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()  # 학습 모드: Dropout·BatchNorm 활성화
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()        # 이전 배치의 기울기 초기화
            logits = model(images)       # 순전파
            loss   = loss_fn(logits, labels)  # 손실 계산
            loss.backward()              # 역전파: 각 파라미터의 기울기 계산
            optimizer.step()             # 파라미터 업데이트

            # 100 스텝마다 현재 손실 출력
            if step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        # 에폭 종료 시 테스트 세트 전체 평가
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # ── 예측 결과 이미지 저장 ─────────────────────────────────────────────
    save_predictions(
        model,
        test_loader,
        device,
        args.output_dir / with_script_prefix(__file__, "mnist_mlp_predictions.png"),
    )


if __name__ == "__main__":
    main()
