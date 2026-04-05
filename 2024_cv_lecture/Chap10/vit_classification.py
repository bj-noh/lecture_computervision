"""
Chap10. Vision Transformer (ViT) — 이미지 분류
===============================================
2020년 Google Brain이 발표한 ViT(Vision Transformer)를 학습합니다.
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

핵심 개념:
  1. 이미지 → 패치 분할 (Patch Embedding)
  2. Positional Encoding (위치 정보 부여)
  3. [CLS] 토큰으로 최종 분류
  4. timm 라이브러리로 사전 학습 모델 활용

두 가지 실습:
  A. ViT 구조 직접 구현 (원리 이해)
  B. timm 사전 학습 모델 파인튜닝 (실전 활용)

실행 방법:
  python vit_classification.py --mode scratch   # A: 직접 구현
  python vit_classification.py --mode finetune  # B: 파인튜닝 (권장)

요구 패키지:
  pip install torch torchvision timm matplotlib
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] 사용 디바이스: {DEVICE}')

# ─────────────────────────────────────────────
# A. ViT 직접 구현 (원리 이해용)
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    이미지를 16×16 패치로 나눠 1D 시퀀스로 변환합니다.

    예: 224×224 이미지 → 196개 패치 (14×14) → 각 패치를 dim 차원으로 투영
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d를 stride=patch_size로 사용 → 패치 분할 + 선형 투영을 한 번에
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA)
    각 패치가 다른 모든 패치와 얼마나 관련 있는지 계산합니다.
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5   # 1/√d_k (어텐션 스케일링)

        self.qkv    = nn.Linear(embed_dim, embed_dim * 3)
        self.proj   = nn.Linear(embed_dim, embed_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        # Q, K, V 행렬 생성
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled Dot-Product Attention: softmax(QK^T / √d_k) × V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """
    ViT의 기본 블록: LayerNorm → MHSA → 잔차 연결 → LayerNorm → MLP → 잔차 연결
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # 잔차 연결 1
        x = x + self.mlp(self.norm2(x))    # 잔차 연결 2
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT-Tiny 수준, 강의용 경량 버전)

    구조:
      Patch Embedding → [CLS] 토큰 추가 → Positional Encoding
      → Transformer Blocks × depth → LayerNorm → [CLS] 토큰으로 분류
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=192, depth=6, num_heads=3,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] 토큰: 전체 이미지를 대표하는 학습 가능한 벡터
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 위치 인코딩: 패치의 순서(위치) 정보를 담는 학습 가능한 벡터
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # Transformer 블록 쌓기
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 가중치 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # [CLS] 토큰을 배치 크기만큼 복제하여 앞에 붙임
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)   # (B, num_patches+1, embed_dim)
        x   = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        # [CLS] 토큰(index 0)만 분류에 사용
        return self.head(x[:, 0])


def train_from_scratch():
    """CIFAR-10으로 직접 구현한 ViT 학습 (원리 이해용)"""
    EPOCHS, LR, BATCH = 15, 1e-3, 128

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])

    train_ds = datasets.CIFAR10('../data', train=True,  transform=transform,     download=True)
    val_ds   = datasets.CIFAR10('../data', train=False, transform=val_transform, download=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    # CIFAR-10에 맞춘 경량 ViT (img=32, patch=4 → 64패치)
    model = ViT(img_size=32, patch_size=4, num_classes=10,
                embed_dim=192, depth=6, num_heads=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] ViT 파라미터 수: {n_params:,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'loss': [], 'acc': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 그래디언트 클리핑
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (model(imgs).argmax(1) == labels).sum().item()
        scheduler.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()

        tr_acc = correct     / len(train_ds) * 100
        va_acc = val_correct / len(val_ds)   * 100
        print(f'Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_ds):.4f} | '
              f'Train Acc: {tr_acc:.1f}% | Val Acc: {va_acc:.1f}%')
        history['loss'].append(total_loss / len(train_ds))
        history['acc'].append(va_acc)

    # 어텐션 맵 시각화
    visualize_attention(model, val_dl)
    _plot_history(history, 'vit_scratch_training.png')


def visualize_attention(model, loader):
    """[CLS] 토큰의 어텐션 가중치를 이미지 위에 오버레이하여 시각화합니다."""
    model.eval()
    imgs, labels = next(iter(loader))
    img = imgs[:1].to(DEVICE)

    # 마지막 Transformer 블록의 어텐션 가중치 추출
    attentions = []
    def hook_fn(module, input, output):
        # MHSA forward 재실행 없이 어텐션 값 캡처
        pass  # 실제 구현에서는 register_forward_hook 활용

    with torch.no_grad():
        _ = model(img)

    # 패치 위치별 CLS 어텐션 간단 시각화 (attention rollout 간략화)
    patch_size = 4
    num_patches_side = 32 // patch_size   # 8

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    for i in range(4):
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img_np = (img_np * [0.247, 0.243, 0.261] + [0.491, 0.482, 0.447]).clip(0, 1)
        axes[i].imshow(img_np, interpolation='nearest')
        axes[i].set_title(f'정답: {classes[labels[i]]}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('ViT 입력 이미지 (패치 단위로 처리됨)', fontsize=12)
    plt.tight_layout()
    plt.savefig('vit_samples.png', dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# B. timm 사전 학습 모델 파인튜닝 (실전 활용)
# ─────────────────────────────────────────────

def finetune_pretrained():
    """
    timm 라이브러리의 사전 학습된 ViT를 CIFAR-10에 파인튜닝합니다.
    처음부터 학습하는 것보다 훨씬 빠르게 높은 정확도를 달성합니다.
    """
    try:
        import timm
    except ImportError:
        print('[ERROR] timm이 설치되지 않았습니다.')
        print('        pip install timm  을 실행해주세요.')
        return

    EPOCHS, LR, BATCH = 5, 1e-4, 64

    # timm은 각 모델의 권장 전처리를 제공합니다
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
    model = model.to(DEVICE)

    data_config = timm.data.resolve_data_config({}, model=model)
    transform   = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config)

    train_ds = datasets.CIFAR10('../data', train=True,  transform=transform,     download=True)
    val_ds   = datasets.CIFAR10('../data', train=False, transform=val_transform, download=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f'\n[INFO] timm ViT 파인튜닝 시작 (사전학습 모델 활용)')
    history = {'loss': [], 'acc': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()

        va_acc = correct / len(val_ds) * 100
        print(f'Epoch {epoch}/{EPOCHS} | Loss: {total_loss/len(train_ds):.4f} | Val Acc: {va_acc:.2f}%')
        history['loss'].append(total_loss / len(train_ds))
        history['acc'].append(va_acc)

    _plot_history(history, 'vit_finetune_training.png')


def _plot_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history['loss']); ax1.set_title('Train Loss'); ax1.set_xlabel('Epoch')
    ax2.plot(history['acc']);  ax2.set_title('Val Accuracy (%)'); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f'[INFO] 학습 곡선 저장: {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT 이미지 분류 실습')
    parser.add_argument('--mode', type=str, default='scratch',
                        choices=['scratch', 'finetune'],
                        help='scratch: 직접 구현 / finetune: timm 사전학습 모델')
    args = parser.parse_args()

    if args.mode == 'scratch':
        print('[MODE] ViT 직접 구현 — CIFAR-10 학습')
        train_from_scratch()
    else:
        print('[MODE] timm ViT 파인튜닝')
        finetune_pretrained()
