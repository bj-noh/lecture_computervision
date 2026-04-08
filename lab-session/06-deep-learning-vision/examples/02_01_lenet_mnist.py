from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from output_naming import with_script_prefix


class LeNet5(nn.Module):
    """
    LeCun et al. (1998)이 제안한 LeNet-5 구조 (MNIST 적용 버전).

    원본 LeNet-5는 32×32 입력을 사용하지만,
    여기서는 MNIST 28×28 이미지에 맞게 조정됨.

    특징 추출부(features) + 분류부(classifier) 2단계 구조:
    ┌─────────────────────────────────────────────────────────┐
    │ features  : Conv → Tanh → AvgPool → Conv → Tanh → AvgPool│
    │ classifier: Flatten → FC(256→120) → Tanh → FC(120→84)   │
    │             → Tanh → FC(84→10)                           │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self) -> None:
        super().__init__()

        # ── 특징 추출부 (Convolutional Layers) ───────────────────────────
        # MLP와 달리 컨볼루션 필터가 이미지의 공간적 패턴(edge, corner 등)을 학습
        self.features = nn.Sequential(
            # C1: 입력 1채널 → 6채널 필터, 5×5 커널
            #     출력 크기: (28-5+1) = 24 → 24×24×6
            nn.Conv2d(1, 6, kernel_size=5),

            # 원본 LeNet-5 활성화 함수: Tanh (현대에는 ReLU로 대체되는 경우 많음)
            nn.Tanh(),

            # S2: 2×2 평균 풀링, stride=2 → 공간 해상도 절반으로 감소
            #     출력 크기: 12×12×6
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: 6채널 → 16채널, 5×5 커널
            #     출력 크기: (12-5+1) = 8 → 8×8×16
            nn.Conv2d(6, 16, kernel_size=5),

            nn.Tanh(),

            # S4: 2×2 평균 풀링 → 출력 크기: 4×4×16
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # ── 분류부 (Fully Connected Layers) ──────────────────────────────
        self.classifier = nn.Sequential(
            # 4×4×16 = 256차원 벡터로 펼침
            nn.Flatten(),

            # F5: 256 → 120
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),

            # F6: 120 → 84
            nn.Linear(120, 84),
            nn.Tanh(),

            # 출력층: 84 → 10 (MNIST 클래스 수)
            # CrossEntropyLoss가 Softmax를 포함하므로 여기선 선형 출력(로짓)
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파: 특징 추출 → 분류."""
        x = self.features(x)       # 컨볼루션으로 공간 피처 추출
        return self.classifier(x)  # 완전연결층으로 클래스 분류


def save_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    """
    테스트 이미지 9장을 3×3 그리드로 시각화하고 예측 결과를 저장.

    각 이미지 위에 정답(gt)과 예측(pred)을 함께 표시.

    Args:
        model:       예측에 사용할 모델
        loader:      테스트 DataLoader
        device:      연산 장치
        output_path: 저장할 PNG 경로
    """
    model.eval()
    # 첫 번째 배치에서 앞 9장만 사용
    images, labels = next(iter(loader))
    images = images[:9].to(device)
    labels = labels[:9]

    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()  # 각 이미지의 예측 클래스

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, image, label, pred in zip(axes.flat, images.cpu(), labels, preds):
        ax.imshow(image.squeeze(0), cmap="gray")           # 채널 차원 제거 후 흑백 표시
        ax.set_title(f"gt={label.item()} pred={pred.item()}")  # 정답 vs 예측
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)  # 메모리 해제


def main() -> None:
    """학습 파이프라인: 인자 파싱 → 데이터 로딩 → 학습 → 평가 → 결과 저장."""

    # ── 커맨드라인 인자 파싱 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train a LeNet-style model on MNIST.")
    parser.add_argument("--epochs",     type=int,   default=1)          # 학습 에폭 수
    parser.add_argument("--batch-size", type=int,   default=128)        # 미니배치 크기
    parser.add_argument("--lr",         type=float, default=1e-3)       # Adam 학습률
    parser.add_argument("--data-root",  type=Path,  default=Path("data"))
    parser.add_argument("--output-dir", type=Path,  default=Path("outputs"))
    args = parser.parse_args()

    # ── 장치·디렉터리 설정 ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 전처리 및 로딩 ────────────────────────────────────────────
    # MNIST 공식 통계(평균=0.1307, 표준편차=0.3081)로 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),                       # PIL 이미지 → [0,1] 텐서
        transforms.Normalize((0.1307,), (0.3081,)),  # 표준화 (z-score)
    ])
    train_set = datasets.MNIST(root=args.data_root, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)   # 학습: 셔플 O
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False)  # 평가: 셔플 X

    # ── 모델·옵티마이저·손실 함수 초기화 ─────────────────────────────────
    model     = LeNet5().to(device)                              # LeNet-5를 지정 장치로 이동
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.CrossEntropyLoss()  # 소프트맥스 + 음의 로그우도 손실

    # ── 학습 루프 ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()  # 학습 모드 (Dropout·BN 활성화)
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()             # 이전 기울기 초기화
            logits = model(images)            # 순전파
            loss   = loss_fn(logits, labels)  # 손실 계산
            loss.backward()                   # 역전파
            optimizer.step()                  # 파라미터 업데이트

            if step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        # ── 에폭 종료 후 테스트 세트 전체 평가 ───────────────────────────
        model.eval()  # 평가 모드
        correct = 0
        total   = 0
        with torch.no_grad():  # 기울기 계산 비활성화 → 메모리·속도 최적화
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds   = model(images).argmax(dim=1)       # 예측 클래스
                correct += (preds == labels).sum().item()   # 정답 수 누적
                total   += images.size(0)

        print(f"epoch={epoch} test_acc={correct / total:.4f}")

    # ── 예측 결과 이미지 저장 ─────────────────────────────────────────────
    save_predictions(
        model,
        test_loader,
        device,
        args.output_dir / with_script_prefix(__file__, "lenet_predictions.png"),
    )


if __name__ == "__main__":
    main()
