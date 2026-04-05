"""
Chap09. 시맨틱 분할 (Semantic Segmentation) — U-Net
====================================================
U-Net 아키텍처를 처음부터 구현하고, 합성 데이터셋으로 학습합니다.
(실제 수업에서는 Oxford-IIIT Pet, PASCAL VOC 등으로 교체 가능)

U-Net 핵심 아이디어:
  - Encoder: 특징 추출 (해상도 감소)
  - Decoder: 해상도 복원 (업샘플링)
  - Skip Connection: Encoder 특징 맵을 Decoder에 직접 연결
    → 위치 정보 손실 방지 (의료 영상 분야에서 처음 제안)

실행 방법:
  python unet_segmentation.py

요구 패키지:
  pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# 0. 하이퍼파라미터
# ─────────────────────────────────────────────
IMG_SIZE    = 128      # 입력 이미지 크기
NUM_CLASSES = 3        # 배경(0), 원(1), 사각형(2)
BATCH_SIZE  = 8
NUM_EPOCHS  = 20
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] 사용 디바이스: {DEVICE}')

# ─────────────────────────────────────────────
# 1. 합성 데이터셋 (도형 분할)
# ─────────────────────────────────────────────
class ShapeDataset(Dataset):
    """
    랜덤 원·사각형이 그려진 합성 이미지와 픽셀별 레이블을 생성합니다.
    클래스: 0=배경, 1=원, 2=사각형

    실제 데이터셋으로 교체하려면:
      torchvision.datasets.VOCSegmentation 또는
      datasets.OxfordIIITPet(target_types='segmentation')
    """
    def __init__(self, size=1000, img_size=128):
        self.size     = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        np.random.seed(idx)  # 재현성
        img   = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        mask  = np.zeros((self.img_size, self.img_size),    dtype=np.long)

        n_shapes = np.random.randint(2, 5)
        for _ in range(n_shapes):
            shape = np.random.choice(['circle', 'rect'])
            color = np.random.rand(3).astype(np.float32)
            cx, cy = np.random.randint(20, self.img_size - 20, 2)
            r = np.random.randint(10, 25)

            if shape == 'circle':
                ys, xs = np.ogrid[:self.img_size, :self.img_size]
                circle_mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
                img[circle_mask] = color
                mask[circle_mask] = 1  # 클래스 1: 원
            else:
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(self.img_size, cx + r), min(self.img_size, cy + r)
                img[y1:y2, x1:x2] = color
                mask[y1:y2, x1:x2] = 2  # 클래스 2: 사각형

        # HWC → CHW, numpy → tensor
        img_tensor  = torch.from_numpy(img.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask)
        return img_tensor, mask_tensor


# ─────────────────────────────────────────────
# 2. U-Net 모델 구현
# ─────────────────────────────────────────────
def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """U-Net 기본 블록: Conv → BN → ReLU → Conv → BN → ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    표준 U-Net 구조
    ─────────────────────────────────────────
    Encoder (수축 경로):
      3 → 64 → 128 → 256 → 512
    Bottleneck:
      512 → 1024
    Decoder (확장 경로):
      1024+512 → 512 → ... → 64+64 → 64
    Output:
      64 → num_classes
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # ── Encoder ──
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = conv_block(512, 1024)

        # ── Decoder ──
        self.up4   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4  = conv_block(1024, 512)   # skip + up → 512+512=1024 입력

        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3  = conv_block(512, 256)

        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2  = conv_block(256, 128)

        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1  = conv_block(128, 64)

        # ── 출력층 ──
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder (skip connection: torch.cat으로 Encoder 특징 맵 합침)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)   # (B, num_classes, H, W)


# ─────────────────────────────────────────────
# 3. IoU (Intersection over Union) 평가 지표
# ─────────────────────────────────────────────
def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int) -> float:
    """
    픽셀 단위 클래스별 IoU를 계산하고 평균(mIoU)을 반환합니다.

    IoU = TP / (TP + FP + FN)
    mIoU = 클래스별 IoU의 평균
    """
    iou_list = []
    pred_flat = pred_mask.view(-1)
    true_flat = true_mask.view(-1)

    for cls in range(num_classes):
        pred_c = (pred_flat == cls)
        true_c = (true_flat == cls)
        intersection = (pred_c & true_c).sum().float()
        union        = (pred_c | true_c).sum().float()
        if union == 0:
            continue
        iou_list.append((intersection / union).item())

    return sum(iou_list) / len(iou_list) if iou_list else 0.0


# ─────────────────────────────────────────────
# 4. 학습
# ─────────────────────────────────────────────
train_dataset = ShapeDataset(size=1000, img_size=IMG_SIZE)
val_dataset   = ShapeDataset(size=200,  img_size=IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

model     = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[INFO] U-Net 파라미터 수: {param_count:,}')

history = {'train_loss': [], 'val_miou': []}

for epoch in range(1, NUM_EPOCHS + 1):
    # ── 학습 ──
    model.train()
    total_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    scheduler.step()
    avg_loss = total_loss / len(train_loader.dataset)

    # ── 검증 (mIoU) ──
    model.eval()
    miou_list = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            for p, m in zip(preds, masks):
                miou_list.append(compute_iou(p.cpu(), m.cpu(), NUM_CLASSES))

    avg_miou = sum(miou_list) / len(miou_list)
    print(f'Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | mIoU: {avg_miou:.4f}')
    history['train_loss'].append(avg_loss)
    history['val_miou'].append(avg_miou)

# ─────────────────────────────────────────────
# 5. 결과 시각화
# ─────────────────────────────────────────────
COLORS = np.array([[0, 0, 0], [0, 120, 255], [255, 80, 0]], dtype=np.uint8)  # 배경/원/사각형

model.eval()
imgs, masks = next(iter(val_loader))
with torch.no_grad():
    preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu()

n_show = 4
fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))
fig.suptitle('U-Net 시맨틱 분할 결과', fontsize=14, fontweight='bold')

for i in range(n_show):
    img_np   = imgs[i].permute(1, 2, 0).numpy().clip(0, 1)
    mask_np  = COLORS[masks[i].numpy()]
    pred_np  = COLORS[preds[i].numpy()]

    axes[i, 0].imshow(img_np);  axes[i, 0].set_title('입력 이미지'); axes[i, 0].axis('off')
    axes[i, 1].imshow(mask_np); axes[i, 1].set_title('정답 마스크'); axes[i, 1].axis('off')
    axes[i, 2].imshow(pred_np); axes[i, 2].set_title('예측 마스크'); axes[i, 2].axis('off')

legend_patches = [
    mpatches.Patch(color='black',       label='배경 (0)'),
    mpatches.Patch(color='royalblue',   label='원 (1)'),
    mpatches.Patch(color='darkorange',  label='사각형 (2)'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('unet_result.png', dpi=150, bbox_inches='tight')
plt.show()
print('[INFO] 결과 저장: unet_result.png')

# 학습 곡선
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history['train_loss']); ax1.set_title('Train Loss'); ax1.set_xlabel('Epoch')
ax2.plot(history['val_miou']);   ax2.set_title('Val mIoU');   ax2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('unet_training.png', dpi=150)
plt.show()
