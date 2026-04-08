"""
02_03_zfnet_stl10.py
────────────────────────────────────────────────────────────────────────────
ZFNet(Zeiler & Fergus, 2013)을 직접 구현하고 STL-10 데이터셋으로 학습합니다.
학습 종료 후 논문의 핵심 기법인 Deconvnet(역합성곱) 시각화를 수행합니다.

ZFNet vs AlexNet 주요 차이점:
  - Conv1: 11×11, s=4  →  7×7, s=2  (더 세밀한 피처 추출)
  - Conv2: 5×5,  s=1  →  5×5, s=2

ZFNet 논문의 핵심 기여 — Deconvnet 시각화:
  순전파: 입력 이미지 → [Conv → ReLU → LRN → MaxPool] × 2 → [Conv×3] → FC
  역투영: 특정 피처맵 → [MaxUnpool → ReLU → ConvTranspose] × L → 픽셀 공간

  MaxPool의 스위치(switch) 위치를 저장해두었다가 MaxUnpool에서 활용함으로써
  어떤 입력 패턴이 해당 피처를 활성화시키는지를 시각화할 수 있음.

실행 예시:
    python 02_03_zfnet_stl10.py --epochs 30
    python 02_03_zfnet_stl10.py --epochs 30 --vis-layer conv2  # 시각화 레이어 지정

출력:
    outputs/zfnet_stl10_history.png          ← 학습 손실·정확도 곡선
    outputs/zfnet_feature_maps.png           ← 각 레이어 피처맵 그리드
    outputs/zfnet_deconvnet_{layer}.png      ← Deconvnet 역투영 결과
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from output_naming import with_script_prefix


# ══════════════════════════════════════════════════════════════════════════
# ZFNet 모델 (시각화 기능 내장)
# ══════════════════════════════════════════════════════════════════════════
class ZFNetVis(nn.Module):
    """
    ZFNet (Zeiler & Fergus, 2013) — Deconvnet 시각화 기능 내장 버전.

    일반 ZFNet과 달리 아래 정보를 forward 시 자동 저장:
      _fmaps        : 각 Conv 레이어 이후의 피처맵  {layer_name: Tensor}
      _pool_indices : 각 MaxPool의 스위치(argmax) 위치
      _pre_pool_sz  : MaxPool 직전 텐서 크기 (Unpool 시 필요)
      _pre_conv_sz  : ConvTranspose2d output_size 결정용

    입력 크기: 224×224 기준 각 레이어 출력 크기
      conv1 : (B, 96,  110, 110)   pool1: (B, 96,  55,  55)
      conv2 : (B, 256,  28,  28)   pool2: (B, 256, 14,  14)
      conv3 : (B, 384,  14,  14)
      conv4 : (B, 384,  14,  14)
      conv5 : (B, 256,  14,  14)   pool3: (B, 256,  7,   7)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ── Block 1 ───────────────────────────────────────────────────────
        # AlexNet 11×11/s=4 → ZFNet 7×7/s=2 : 초기 aliasing 감소
        self.conv1 = nn.Conv2d(3,   96,  kernel_size=7, stride=2, padding=1)
        self.lrn1  = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        # return_indices=True : MaxPool 스위치 위치 반환 (Deconvnet 역투영에 필요)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        # ── Block 2 ───────────────────────────────────────────────────────
        # stride=2 (AlexNet은 stride=1)
        self.conv2 = nn.Conv2d(96,  256, kernel_size=5, stride=2, padding=2)
        self.lrn2  = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        # ── Block 3~5 (AlexNet과 동일) ────────────────────────────────────
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        self.avgpool    = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # 시각화용 저장소 (forward 시 자동 갱신)
        self._fmaps:       dict[str, torch.Tensor] = {}
        self._pool_idx:    dict[str, torch.Tensor] = {}
        self._pre_pool_sz: dict[str, torch.Size]   = {}

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Block 1 ───────────────────────────────────────────────────────
        x = F.relu(self.conv1(x))
        self._fmaps["conv1"] = x.detach().clone()               # 피처맵 저장
        x = self.lrn1(x)
        self._pre_pool_sz["pool1"] = x.shape                    # (B,96,110,110)
        x, idx = self.pool1(x)
        self._pool_idx["pool1"] = idx                           # 스위치 저장

        # ── Block 2 ───────────────────────────────────────────────────────
        x = F.relu(self.conv2(x))
        self._fmaps["conv2"] = x.detach().clone()
        x = self.lrn2(x)
        self._pre_pool_sz["pool2"] = x.shape                    # (B,256,28,28)
        x, idx = self.pool2(x)
        self._pool_idx["pool2"] = idx

        # ── Block 3~5 ─────────────────────────────────────────────────────
        x = F.relu(self.conv3(x))
        self._fmaps["conv3"] = x.detach().clone()
        x = F.relu(self.conv4(x))
        self._fmaps["conv4"] = x.detach().clone()
        x = F.relu(self.conv5(x))
        self._fmaps["conv5"] = x.detach().clone()
        self._pre_pool_sz["pool3"] = x.shape                    # (B,256,14,14)
        x, idx = self.pool3(x)
        self._pool_idx["pool3"] = idx

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════
# Deconvnet 역투영 (논문 핵심 기법)
# ══════════════════════════════════════════════════════════════════════════
def deconvnet_project(
    model:      ZFNetVis,
    layer_name: str,
    sample_idx: int = 0,
    n_top:      int = 1,
    device:     torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    ZFNet 논문의 Deconvnet 역투영.

    선택한 레이어(layer_name)의 피처맵에서 가장 강하게 활성화된
    상위 n_top개 뉴런만 남기고 역합성곱을 통해 픽셀 공간으로 투영.

    역투영 과정 (예: conv5 → 입력):
      pool3 스위치 → MaxUnpool3 → ReLU
      → ConvTranspose(conv5 가중치)
      → ConvTranspose(conv4 가중치)
      → ConvTranspose(conv3 가중치)
      → pool2 스위치 → MaxUnpool2 → ReLU
      → ConvTranspose(conv2 가중치, s=2)
      → pool1 스위치 → MaxUnpool1 → ReLU
      → ConvTranspose(conv1 가중치, s=2)
      → 224×224×3 픽셀 공간

    Args:
        model:      ZFNetVis 인스턴스 (forward 한 번 이상 실행된 상태)
        layer_name: 역투영 시작 레이어 ('conv1'~'conv5')
        sample_idx: 배치 내 샘플 인덱스
        n_top:      남길 최대 활성화 수
        device:     연산 장치

    Returns:
        pixel_proj: (3, H, W) 픽셀 공간 역투영 텐서
    """
    # 해당 레이어 피처맵 가져오기 (단일 샘플)
    fmap = model._fmaps[layer_name][sample_idx:sample_idx+1].clone().to(device)  # (1,C,H,W)

    # ── 상위 n_top 뉴런만 남기기 ─────────────────────────────────────────
    flat   = fmap.abs().view(-1)
    if n_top < flat.numel():
        threshold = flat.topk(n_top).values[-1]
        fmap      = fmap * (fmap.abs() >= threshold).float()
    x = fmap

    # ── Unpooler 준비 ────────────────────────────────────────────────────
    unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1).to(device)
    unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1).to(device)
    unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1).to(device)

    layer_order = ["conv5", "conv4", "conv3", "conv2", "conv1"]
    start_idx   = layer_order.index(layer_name)

    # ── 역방향 투영 ──────────────────────────────────────────────────────
    for layer in layer_order[start_idx:]:
        if layer == "conv5":
            # pool3 역투영
            idx3 = model._pool_idx["pool3"][sample_idx:sample_idx+1].to(device)
            sz3  = model._pre_pool_sz["pool3"]
            x    = unpool3(x, idx3, output_size=(1, sz3[1], sz3[2], sz3[3]))
            x    = F.relu(x)
            # conv5 역합성곱
            x = F.conv_transpose2d(x, model.conv5.weight, stride=1, padding=1)
            x = F.relu(x)

        elif layer == "conv4":
            x = F.conv_transpose2d(x, model.conv4.weight, stride=1, padding=1)
            x = F.relu(x)

        elif layer == "conv3":
            x = F.conv_transpose2d(x, model.conv3.weight, stride=1, padding=1)
            x = F.relu(x)
            # pool2 역투영
            idx2 = model._pool_idx["pool2"][sample_idx:sample_idx+1].to(device)
            sz2  = model._pre_pool_sz["pool2"]
            x    = unpool2(x, idx2, output_size=(1, sz2[1], sz2[2], sz2[3]))
            x    = F.relu(x)

        elif layer == "conv2":
            # conv2: stride=2 → ConvTranspose2d에서도 stride=2
            x = F.conv_transpose2d(x, model.conv2.weight, stride=2, padding=2)
            x = F.relu(x)
            # pool1 역투영
            idx1 = model._pool_idx["pool1"][sample_idx:sample_idx+1].to(device)
            sz1  = model._pre_pool_sz["pool1"]
            x    = unpool1(x, idx1, output_size=(1, sz1[1], sz1[2], sz1[3]))
            x    = F.relu(x)

        elif layer == "conv1":
            # conv1: stride=2, padding=1, kernel=7 → output_padding=1 필요
            x = F.conv_transpose2d(x, model.conv1.weight, stride=2, padding=1, output_padding=1)
            # ReLU 없음 (픽셀 공간 최종 재구성)

    return x.squeeze(0)  # (3, H, W)


# ══════════════════════════════════════════════════════════════════════════
# Feature Map 시각화
# ══════════════════════════════════════════════════════════════════════════
def visualize_feature_maps(
    model:       ZFNetVis,
    image:       torch.Tensor,
    device:      torch.device,
    output_path: Path,
    n_filters:   int = 16,
) -> None:
    """
    각 Conv 레이어의 피처맵(activation map)을 그리드로 시각화.

    피처맵: 입력 이미지가 각 Conv 필터를 통과한 후의 활성화 값.
    필터마다 다른 패턴(edge, 색상, 텍스처 등)에 반응하는 것을 확인 가능.

    Args:
        model:     ZFNetVis 인스턴스
        image:     입력 이미지 텐서 (1, 3, H, W)
        device:    연산 장치
        output_path: 저장 경로
        n_filters: 레이어당 표시할 최대 필터 수
    """
    model.eval()
    with torch.no_grad():
        model(image.to(device))

    layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    layer_info  = {
        "conv1": "Conv1 (96ch, 110×110)",
        "conv2": "Conv2 (256ch, 28×28)",
        "conv3": "Conv3 (384ch, 14×14)",
        "conv4": "Conv4 (384ch, 14×14)",
        "conv5": "Conv5 (256ch, 14×14)",
    }

    fig, axes = plt.subplots(len(layer_names), n_filters,
                             figsize=(n_filters * 1.2, len(layer_names) * 1.5))
    fig.suptitle("ZFNet Feature Maps (Activation Maps per Conv Layer)",
                 fontsize=13, fontweight="bold")

    for row, lname in enumerate(layer_names):
        fmap    = model._fmaps[lname][0]          # (C, H, W) — 첫 번째 샘플
        n_show  = min(n_filters, fmap.shape[0])

        for col in range(n_filters):
            ax = axes[row][col]
            if col < n_show:
                fm = fmap[col].cpu().numpy()
                # 0~1 정규화 (각 채널별)
                fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                ax.imshow(fm, cmap="viridis")
                ax.axis("off")
            else:
                ax.axis("off")

        # 행 레이블 (가장 왼쪽)
        axes[row][0].set_title(layer_info[lname], fontsize=7, loc="left", pad=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Feature Map 시각화 저장: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# Deconvnet 시각화
# ══════════════════════════════════════════════════════════════════════════
def visualize_deconvnet(
    model:       ZFNetVis,
    images:      torch.Tensor,
    labels:      torch.Tensor,
    class_names: list[str],
    device:      torch.device,
    output_dir:  Path,
    layers:      list[str] | None = None,
    n_top:       int = 9,
) -> None:
    """
    Zeiler & Fergus (2013) Deconvnet 역투영 시각화.

    선택한 레이어에서 가장 강하게 활성화된 뉴런들을 역투영하여
    어떤 입력 패턴이 해당 피처를 활성화시키는지 시각화.

    ┌──────────────────────────────────────────────────────┐
    │  원본 이미지          Deconvnet 역투영              │
    │  ───────────       ─────────────────────────────    │
    │  [이미지 A]   →   conv1: 저수준 edge 패턴          │
    │               →   conv2: 텍스처/코너 패턴           │
    │               →   conv5: 고수준 의미 패턴           │
    └──────────────────────────────────────────────────────┘

    Args:
        layers:  역투영할 레이어 목록 (None이면 conv1, conv2, conv5)
        n_top:   각 레이어에서 유지할 최대 활성화 수
    """
    if layers is None:
        layers = ["conv1", "conv2", "conv5"]

    model.eval()
    with torch.no_grad():
        model(images.to(device))

    for layer_name in layers:
        n_samples = min(images.shape[0], 4)
        cols = n_samples + 1   # projection + source image
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3.5))

        fig.suptitle(
            f"ZFNet Deconvnet - {layer_name.upper()} Projection\n"
            f"(Pixel-space projection with Top-{n_top} activations kept)",
            fontsize=11, fontweight="bold"
        )

        # 정규화 역변환 (시각화용)
        mean = torch.tensor([0.4467, 0.4398, 0.4066]).view(3, 1, 1)
        std  = torch.tensor([0.2603, 0.2566, 0.2713]).view(3, 1, 1)

        # First column: source image (color)
        orig_grid = images[0].cpu() * std + mean
        orig_grid = orig_grid.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0].imshow(orig_grid)
        axes[0].set_title(f"Source\n({class_names[labels[0].item()]})",
                          fontsize=9, fontweight="bold")
        axes[0].axis("off")

        # Remaining columns: per-sample deconvnet projection
        for s in range(n_samples):
            try:
                proj = deconvnet_project(model, layer_name,
                                         sample_idx=s, n_top=n_top, device=device)
                proj_np = proj.cpu().detach()
                # 절댓값으로 강도 표현, 정규화
                proj_np = proj_np.abs().sum(dim=0).numpy()  # (H, W) — 채널 합산
                proj_np = (proj_np - proj_np.min()) / (proj_np.max() - proj_np.min() + 1e-8)

                ax = axes[s + 1]
                im = ax.imshow(proj_np, cmap="hot")
                ax.set_title(f"Sample {s+1}\n({class_names[labels[s].item()]})",
                             fontsize=9)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            except Exception as e:
                axes[s + 1].set_title(f"Sample {s+1}\n(Error: {e})", fontsize=7)
                axes[s + 1].axis("off")

        fig.tight_layout()
        out = output_dir / with_script_prefix(__file__, f"zfnet_deconvnet_{layer_name}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"Deconvnet 역투영 저장: {out}")


# ══════════════════════════════════════════════════════════════════════════
# 유틸리티: 학습·평가 함수
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_correct = total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits         = model(images)
            total_loss    += loss_fn(logits, labels).item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count   += images.size(0)
    return total_loss / total_count, total_correct / total_count


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = total_correct = total_count = 0
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


def plot_history(history, epochs, title, output_path):
    x = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax1.plot(x, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    ax1.plot(x, history["test_loss"],  label="Test Loss",  color="#2980b9", linewidth=2, linestyle="--")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(x, [a*100 for a in history["train_acc"]], label="Train Acc", color="#e74c3c", linewidth=2)
    ax2.plot(x, [a*100 for a in history["test_acc"]],  label="Test Acc",  color="#2980b9", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"그래프 저장 완료: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="ZFNet + Deconvnet Visualization on STL-10.")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data-root",  type=Path,  default=Path("data"))
    parser.add_argument("--output-dir", type=Path,  default=Path("outputs"))
    parser.add_argument("--vis-layer",  type=str,   default=None,
                        help="Deconvnet 시각화 레이어 (conv1~conv5, 기본: conv1+conv2+conv5)")
    parser.add_argument("--n-top",      type=int,   default=9,
                        help="역투영 시 유지할 최대 활성화 수")
    parser.add_argument("--skip-train", action="store_true",
                        help="학습 생략하고 시각화만 실행 (체크포인트 있을 때)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"사용 장치: {device}")
    print(f"데이터셋: STL-10 (96×96→224×224, 10 classes)")
    print(f"모델: ZFNet (Zeiler & Fergus, 2013) with Deconvnet visualization\n")

    # ── 데이터 준비 ───────────────────────────────────────────────────────
    normalize = transforms.Normalize(
        mean=(0.4467, 0.4398, 0.4066),
        std =(0.2603, 0.2566, 0.2713),
    )
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.STL10(root=args.data_root, split="train", download=True, transform=train_transform)
    test_set  = datasets.STL10(root=args.data_root, split="test",  download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"훈련: {len(train_set):,}장  /  테스트: {len(test_set):,}장")
    print(f"STL-10 클래스: {train_set.classes}\n")

    # ── 모델 초기화 ───────────────────────────────────────────────────────
    model      = ZFNetVis(num_classes=10).to(device)
    ckpt_path  = args.output_dir / "zfnet_stl10.pt"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ZFNet 총 파라미터: {total_params:,}\n")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    if not args.skip_train:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_fn   = nn.CrossEntropyLoss()

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "test_loss":  [], "test_acc":  [],
        }

        print(f"{'Epoch':>6} {'LR':>8} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9}")
        print("-" * 62)

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_loss,  test_acc  = evaluate(model, test_loader, loss_fn, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            flag = " ★" if test_acc > best_acc else ""
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), ckpt_path)  # 최고 성능 체크포인트 저장

            print(f"{epoch:>6d} {lr:>8.2e} {train_loss:>11.4f} "
                  f"{train_acc*100:>9.2f}% {test_loss:>10.4f} {test_acc*100:>8.2f}%{flag}")

        best_epoch = history["test_acc"].index(max(history["test_acc"])) + 1
        print(f"\n최고 테스트 정확도: {best_acc*100:.2f}% (epoch {best_epoch})")
        print(f"체크포인트 저장: {ckpt_path}")

        plot_history(history, args.epochs, "ZFNet on STL-10 (Scratch, SGD+Momentum)",
                     args.output_dir / with_script_prefix(__file__, "zfnet_stl10_history.png"))

    elif ckpt_path.exists():
        # 체크포인트 로드
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"체크포인트 로드: {ckpt_path}")

    # ══════════════════════════════════════════════════════════════════════
    # 시각화 파이프라인
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("시각화 시작")
    print("=" * 62)

    # 시각화용 배치 가져오기 (test_loader의 첫 번째 배치)
    vis_images, vis_labels = next(iter(test_loader))
    vis_images = vis_images[:8]   # 최대 8장
    vis_labels = vis_labels[:8]

    # ── 1. Feature Map 시각화 ─────────────────────────────────────────────
    print("\n[1/2] Feature Map 시각화 중...")
    model.eval()
    with torch.no_grad():
        model(vis_images[:1].to(device))   # 단일 이미지로 피처맵 추출

    visualize_feature_maps(
        model       = model,
        image       = vis_images[:1],
        device      = device,
        output_path = args.output_dir / with_script_prefix(__file__, "zfnet_feature_maps.png"),
        n_filters   = 16,
    )

    # ── 2. Deconvnet 역투영 시각화 ────────────────────────────────────────
    print("\n[2/2] Deconvnet 역투영 시각화 중...")
    # 배치 전체로 forward (pool indices 저장)
    model.eval()
    with torch.no_grad():
        model(vis_images.to(device))

    vis_layers = [args.vis_layer] if args.vis_layer else ["conv1", "conv2", "conv5"]
    visualize_deconvnet(
        model       = model,
        images      = vis_images,
        labels      = vis_labels,
        class_names = list(test_set.classes),
        device      = device,
        output_dir  = args.output_dir,
        layers      = vis_layers,
        n_top       = args.n_top,
    )

    print("\n모든 시각화 완료!")
    print(f"저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
