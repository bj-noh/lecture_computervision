"""
Chap11. 확산 모델 (Diffusion Model) — DDPM
==========================================
2020년 Ho et al. "Denoising Diffusion Probabilistic Models (DDPM)"
핵심 원리를 이해하기 위한 경량 구현입니다.

핵심 아이디어:
  - Forward Process: 이미지에 Gaussian 노이즈를 T 단계에 걸쳐 점진적으로 추가
  - Reverse Process: 노이즈로부터 이미지를 단계적으로 복원 (노이즈 예측 U-Net)
  - 학습 목표: 각 타임스텝 t에서 추가된 노이즈 ε를 예측

GAN과의 차이:
  GAN:      Generator가 이미지를 한 번에 생성
  DDPM:     T번의 작은 노이즈 제거 단계를 거쳐 이미지 생성
  장점:     학습 안정적, mode collapse 없음
  단점:     생성 속도 느림 (개선판: DDIM, 잠재 확산 모델 LDM/Stable Diffusion)

실행 방법:
  python ddpm_diffusion.py

요구 패키지:
  pip install torch torchvision matplotlib tqdm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
IMG_SIZE   = 32
CHANNELS   = 1        # Fashion-MNIST (흑백)
T          = 300      # 확산 타임스텝 수 (논문: 1000, 실습용으로 축소)
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 2e-4
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] 사용 디바이스: {DEVICE}')
print(f'[INFO] 확산 타임스텝 T = {T}')

# ─────────────────────────────────────────────
# 1. 노이즈 스케줄 (Beta Schedule)
# ─────────────────────────────────────────────
def get_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule: β_1, β_2, ..., β_T
    β가 클수록 노이즈를 더 많이 추가합니다.
    """
    return torch.linspace(beta_start, beta_end, T)  # (T,)


betas  = get_beta_schedule(T).to(DEVICE)
alphas = 1.0 - betas                                # α_t = 1 - β_t
alphas_cumprod = torch.cumprod(alphas, dim=0)       # ᾱ_t = ∏α_i (i=1..t)

# 시각화: 타임스텝에 따른 신호 vs 노이즈 비율
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(alphas_cumprod.cpu().numpy(), label='ᾱ_t (신호 비율)')
ax.plot(1 - alphas_cumprod.cpu().numpy(), label='1-ᾱ_t (노이즈 비율)')
ax.set_xlabel('타임스텝 t'); ax.set_ylabel('비율')
ax.set_title('Beta Schedule: 시간에 따른 신호/노이즈 변화')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('beta_schedule.png', dpi=120)
plt.close()
print('[INFO] beta_schedule.png 저장 완료')

# ─────────────────────────────────────────────
# 2. Forward Process (노이즈 추가)
# ─────────────────────────────────────────────
def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
    """
    Forward process: x_0에서 x_t를 한 번에 샘플링
    수식: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε  (ε ~ N(0,I))

    closed-form 덕분에 임의의 타임스텝 t를 바로 계산할 수 있습니다.
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_acp    = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1macp  = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_acp * x0 + sqrt_1macp * noise


# ─────────────────────────────────────────────
# 3. 노이즈 예측 U-Net (경량 버전)
# ─────────────────────────────────────────────
class SinusoidalPositionEmbedding(nn.Module):
    """
    타임스텝 t를 연속적인 벡터로 인코딩합니다 (Transformer의 위치 인코딩과 동일 원리).
    모델이 현재 어떤 노이즈 레벨인지 알 수 있게 해줍니다.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


import math

class ResBlock(nn.Module):
    """타임스텝 임베딩을 조건으로 받는 잔차 블록"""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv1    = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.conv2    = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return h + self.shortcut(x)


class DiffusionUNet(nn.Module):
    """
    DDPM의 노이즈 예측 네트워크 (경량 U-Net)
    입력: 노이즈 이미지 x_t + 타임스텝 t
    출력: 예측 노이즈 ε̂ (입력과 같은 크기)
    """
    def __init__(self, channels=1, base_dim=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Encoder
        self.enc1 = ResBlock(channels,    base_dim,     time_dim)
        self.enc2 = ResBlock(base_dim,    base_dim * 2, time_dim)
        self.enc3 = ResBlock(base_dim*2,  base_dim * 4, time_dim)
        self.down  = nn.MaxPool2d(2)

        # Bottleneck
        self.bot   = ResBlock(base_dim * 4, base_dim * 4, time_dim)

        # Decoder
        self.up3   = nn.ConvTranspose2d(base_dim*4, base_dim*2, 2, stride=2)
        self.dec3  = ResBlock(base_dim*4, base_dim*2, time_dim)
        self.up2   = nn.ConvTranspose2d(base_dim*2, base_dim,   2, stride=2)
        self.dec2  = ResBlock(base_dim*2, base_dim,   time_dim)

        self.out   = nn.Conv2d(base_dim, channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)          # 타임스텝 임베딩

        e1 = self.enc1(x,           t_emb)  # 32×32
        e2 = self.enc2(self.down(e1), t_emb)  # 16×16
        e3 = self.enc3(self.down(e2), t_emb)  # 8×8

        b  = self.bot(self.down(e3), t_emb)   # 4×4

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)

        return self.out(d2)


# ─────────────────────────────────────────────
# 4. 데이터 로드 (Fashion-MNIST)
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),   # [-1, 1] 정규화
])

train_ds = datasets.FashionMNIST('../data', train=True,  transform=transform, download=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f'[INFO] Fashion-MNIST 학습 데이터: {len(train_ds):,}장')

# ─────────────────────────────────────────────
# 5. 학습
# ─────────────────────────────────────────────
model     = DiffusionUNet(channels=CHANNELS).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[INFO] 모델 파라미터: {n_params:,}')

losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for x0, _ in tqdm(train_dl, desc=f'Epoch {epoch:02d}/{EPOCHS}', leave=False):
        x0 = x0.to(DEVICE)
        B  = x0.size(0)

        # 랜덤 타임스텝 t ~ Uniform(0, T-1)
        t     = torch.randint(0, T, (B,), device=DEVICE)
        noise = torch.randn_like(x0)

        # Forward: 노이즈 추가
        x_t = q_sample(x0, t, noise)

        # 예측 노이즈와 실제 노이즈의 MSE Loss (Simple Loss)
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_dl)
    losses.append(avg_loss)
    print(f'Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.5f}')

    # 10 에폭마다 샘플 생성
    if epoch % 10 == 0:
        sample_images(model, epoch)

# ─────────────────────────────────────────────
# 6. Reverse Process (이미지 생성)
# ─────────────────────────────────────────────
@torch.no_grad()
def p_sample(model, x_t: torch.Tensor, t: int):
    """
    Reverse process 한 스텝: x_t → x_{t-1}
    수식: x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε̂) + σ_t · z
    """
    beta_t  = betas[t]
    alpha_t = alphas[t]
    acp_t   = alphas_cumprod[t]

    t_tensor = torch.full((x_t.size(0),), t, device=DEVICE, dtype=torch.long)
    pred_noise = model(x_t, t_tensor)

    # 평균 계산
    coef      = beta_t / (1 - acp_t).sqrt()
    mean      = (1 / alpha_t.sqrt()) * (x_t - coef * pred_noise)

    if t == 0:
        return mean
    else:
        # 분산: 단순히 β_t 사용
        noise = torch.randn_like(x_t)
        return mean + beta_t.sqrt() * noise


@torch.no_grad()
def generate_samples(model, n_samples=16):
    """순수 Gaussian 노이즈에서 시작해 T → 0 역방향으로 이미지 생성"""
    model.eval()
    x = torch.randn(n_samples, CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for t in tqdm(reversed(range(T)), desc='생성 중', total=T, leave=False):
        x = p_sample(model, x, t)
    return x


def sample_images(model, epoch):
    """생성된 이미지를 격자 형태로 저장"""
    samples = generate_samples(model, n_samples=16)
    samples = (samples.clamp(-1, 1) + 1) / 2   # [0, 1]로 변환

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'DDPM 생성 샘플 (Epoch {epoch})', fontsize=13)
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'ddpm_epoch{epoch:03d}.png', dpi=120)
    plt.close()
    print(f'[INFO] 생성 샘플 저장: ddpm_epoch{epoch:03d}.png')


# ─────────────────────────────────────────────
# 7. Forward process 시각화 (노이즈 추가 과정)
# ─────────────────────────────────────────────
def visualize_forward_process():
    """원본 이미지에 단계적으로 노이즈가 추가되는 과정을 보여줍니다."""
    sample_img, _ = train_ds[0]
    x0 = sample_img.unsqueeze(0).to(DEVICE)

    timesteps = [0, 50, 100, 150, 200, 250, 299]
    fig, axes = plt.subplots(1, len(timesteps), figsize=(14, 3))
    fig.suptitle('Forward Process: 점진적 노이즈 추가', fontsize=12)

    for ax, t_val in zip(axes, timesteps):
        t_tensor = torch.tensor([t_val], device=DEVICE)
        x_t = q_sample(x0, t_tensor)
        img = x_t[0, 0].cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f't = {t_val}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('forward_process.png', dpi=130)
    plt.show()
    print('[INFO] forward_process.png 저장 완료')


# 학습 손실 곡선
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title('DDPM 학습 손실 (MSE)')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ddpm_loss.png', dpi=120)
plt.show()

visualize_forward_process()
print('\n[INFO] 최종 샘플 생성 중...')
sample_images(model, epoch=EPOCHS)
print('[완료] 모든 결과가 저장되었습니다.')
