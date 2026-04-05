"""
Chap13. GAN (Generative Adversarial Network) — DCGAN
=====================================================
Goodfellow et al. 2014 / Radford et al. 2015 (DCGAN)

핵심 개념:
  - Generator(G):     잠재 벡터 z → 가짜 이미지 생성
  - Discriminator(D): 진짜/가짜 이미지 판별
  - 적대적 학습:      G는 D를 속이려 하고, D는 G를 꿰뚫으려 함
  - 내쉬 균형:        G 생성 이미지가 완벽해지면 D는 50% 확률로 판별

GAN vs DDPM (Chap11 비교):
  GAN:  빠른 생성, 학습 불안정 (mode collapse 위험)
  DDPM: 학습 안정적, 생성 품질 높음, 속도 느림

주요 수정 사항 (기존 버그 수정 + 현대화):
  [BUG FIX] D.trianable → 올바른 PyTorch API로 교체
  [BUG FIX] gray2 변환 시 img1 대신 img2 사용 오류 제거
  [MODERN]  구식 keras/tensorflow 혼용 → 순수 PyTorch로 통일
  [MODERN]  DCGAN 논문 권장 가중치 초기화 추가
  [MODERN]  잠재 공간 보간(Interpolation) 시각화 추가

실행 방법:
  python gan.py

요구 패키지:
  pip install torch torchvision matplotlib tqdm
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0. 하이퍼파라미터
# ─────────────────────────────────────────────
ZDIM       = 100
IMG_SIZE   = 28
CHANNELS   = 1       # Fashion-MNIST 흑백
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 0.0002
BETA1      = 0.5     # Adam β1 (DCGAN 논문 권장)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] 사용 디바이스: {DEVICE}')

# ─────────────────────────────────────────────
# 1. 데이터 로드 (Fashion-MNIST — Bag 클래스)
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),   # [-1, 1] 정규화 (tanh 출력과 맞춤)
])

dataset     = datasets.FashionMNIST('../data', train=True, transform=transform, download=True)
bag_indices = [i for i, (_, label) in enumerate(dataset) if label == 8]
bag_subset  = torch.utils.data.Subset(dataset, bag_indices)
loader      = DataLoader(bag_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f'[INFO] Bag(가방) 클래스 데이터: {len(bag_subset):,}장')

# ─────────────────────────────────────────────
# 2. 모델 정의
# ─────────────────────────────────────────────
def weights_init(m):
    """DCGAN 논문 권장 가중치 초기화"""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    G: z(100차원) → 28×28 이미지
    ConvTranspose2d(전치 합성곱)로 해상도를 단계적으로 키움

    z(100) → Dense → 7×7×256 → UpSample → 14×14 → UpSample → 28×28
    """
    def __init__(self, zdim=100, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 7 * 7 * 256, bias=False),
            nn.Unflatten(1, (256, 7, 7)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 7×7 → 14×14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 14×14 → 28×28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Tanh(),   # 출력 범위 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    D: 28×28 이미지 → 진짜(1) / 가짜(0) 확률
    LeakyReLU 사용 — 음수 기울기를 살려 그래디언트 소실 방지
    """
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # 28×28 → 14×14  (첫 레이어 BN 없음)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 14×14 → 7×7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 7×7 → 4×4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


G = Generator(ZDIM, CHANNELS).to(DEVICE)
D = Discriminator(CHANNELS).to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)

print(f'[INFO] Generator 파라미터: {sum(p.numel() for p in G.parameters()):,}')
print(f'[INFO] Discriminator 파라미터: {sum(p.numel() for p in D.parameters()):,}')

# ─────────────────────────────────────────────
# 3. 손실 함수 / 옵티마이저
# ─────────────────────────────────────────────
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

# 학습 진행 평가용 고정 노이즈 (매 에폭 같은 z로 비교)
fixed_z = torch.randn(16, ZDIM, device=DEVICE)

# ─────────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────────
history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}

for epoch in range(1, EPOCHS + 1):
    d_losses, g_losses, d_reals, d_fakes = [], [], [], []

    for real_imgs, _ in tqdm(loader, desc=f'Epoch {epoch:02d}/{EPOCHS}', leave=False):
        real_imgs = real_imgs.to(DEVICE)
        B = real_imgs.size(0)

        # ── Discriminator 학습 ──────────────────
        D.zero_grad()

        out_real  = D(real_imgs).view(-1)
        loss_real = criterion(out_real, torch.ones(B, device=DEVICE))

        z         = torch.randn(B, ZDIM, device=DEVICE)
        fake_imgs = G(z).detach()   # G 업데이트 차단
        out_fake  = D(fake_imgs).view(-1)
        loss_fake = criterion(out_fake, torch.zeros(B, device=DEVICE))

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # ── Generator 학습 ──────────────────────
        G.zero_grad()

        z         = torch.randn(B, ZDIM, device=DEVICE)
        fake_imgs = G(z)
        out_g     = D(fake_imgs).view(-1)
        loss_G    = criterion(out_g, torch.ones(B, device=DEVICE))  # D를 속이는 게 목표
        loss_G.backward()
        opt_G.step()

        d_losses.append(loss_D.item())
        g_losses.append(loss_G.item())
        d_reals.append(out_real.mean().item())
        d_fakes.append(out_g.mean().item())

    avg_d, avg_g   = np.mean(d_losses), np.mean(g_losses)
    avg_dr, avg_df = np.mean(d_reals),  np.mean(d_fakes)
    history['d_loss'].append(avg_d)
    history['g_loss'].append(avg_g)
    history['d_real'].append(avg_dr)
    history['d_fake'].append(avg_df)

    print(f'Epoch {epoch:02d}/{EPOCHS} | D Loss: {avg_d:.4f}  G Loss: {avg_g:.4f} | '
          f'D(real): {avg_dr:.3f}  D(G(z)): {avg_df:.3f}')

    # 10 에폭마다 샘플 이미지 저장
    if epoch % 10 == 0:
        G.eval()
        with torch.no_grad():
            samples = (G(fixed_z) + 1) / 2   # [-1,1] → [0,1]
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        fig.suptitle(f'GAN 생성 샘플 (Epoch {epoch})', fontsize=11)
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'gan_epoch{epoch:03d}.png', dpi=120)
        plt.close()
        print(f'  → gan_epoch{epoch:03d}.png 저장')
        G.train()

# ─────────────────────────────────────────────
# 5. 학습 곡선 시각화
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['d_loss'], label='D Loss')
ax1.plot(history['g_loss'], label='G Loss')
ax1.set_title('GAN 학습 손실'); ax1.set_xlabel('Epoch')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history['d_real'], label='D(real) 진짜 판별 확률')
ax2.plot(history['d_fake'], label='D(G(z)) 가짜 판별 확률')
ax2.axhline(0.5, color='gray', linestyle='--', label='이상적 균형 (0.5)')
ax2.set_title('Discriminator 출력'); ax2.set_xlabel('Epoch')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gan_training.png', dpi=150)
plt.show()
print('[INFO] 학습 곡선 저장: gan_training.png')

# ─────────────────────────────────────────────
# 6. 잠재 공간 보간 (Latent Space Interpolation)
# ─────────────────────────────────────────────
print('\n[INFO] 잠재 공간 보간 시각화...')
G.eval()
z1 = torch.randn(1, ZDIM, device=DEVICE)
z2 = torch.randn(1, ZDIM, device=DEVICE)

n_steps = 10
fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))
fig.suptitle('잠재 공간 보간: z1 → z2 (두 이미지 사이의 연속적 변화)', fontsize=11)

with torch.no_grad():
    for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
        z_interp = (1 - alpha) * z1 + alpha * z2
        img = ((G(z_interp)[0, 0] + 1) / 2).cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'α={alpha:.1f}', fontsize=8)
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('latent_interpolation.png', dpi=130)
plt.show()
print('[INFO] 저장: latent_interpolation.png')
