"""
Chap07. CNN & 전이학습 (Transfer Learning)
========================================
사전 학습된 ResNet18을 CIFAR-10 데이터셋에 파인튜닝합니다.

핵심 개념:
  - Feature Extractor vs Fine-tuning 차이
  - 학습률 스케줄러 (StepLR)
  - 학습 곡선 시각화

실행 방법:
  python transfer_learning.py

요구 패키지:
  pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# ─────────────────────────────────────────────
# 0. 하이퍼파라미터 설정
# ─────────────────────────────────────────────
BATCH_SIZE  = 128
NUM_EPOCHS  = 10
LR          = 1e-3        # 분류기 헤드 학습률
LR_BACKBONE = 1e-4        # 백본(ResNet) 학습률 (더 작게 설정)
NUM_CLASSES = 10          # CIFAR-10 클래스 수
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] 사용 디바이스: {DEVICE}')

# ─────────────────────────────────────────────
# 1. 데이터 준비
# ─────────────────────────────────────────────
# ImageNet으로 학습된 모델에 맞춰 정규화 (mean/std는 ImageNet 기준)
train_transform = transforms.Compose([
    transforms.Resize(224),               # ResNet 입력 크기
    transforms.RandomHorizontalFlip(),    # 데이터 증강: 좌우 반전
    transforms.RandomCrop(224, padding=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 색상 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet 평균
                         std =[0.229, 0.224, 0.225]),       # ImageNet 표준편차
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# CIFAR-10 자동 다운로드 (../data 폴더에 저장)
train_dataset = datasets.CIFAR10(root='../data', train=True,  transform=train_transform, download=True)
val_dataset   = datasets.CIFAR10(root='../data', train=False, transform=val_transform,   download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASS_NAMES = train_dataset.classes  # ['airplane', 'automobile', ...]
print(f'[INFO] 학습 데이터: {len(train_dataset)}장, 검증 데이터: {len(val_dataset)}장')

# ─────────────────────────────────────────────
# 2. 모델 구성: ResNet18 파인튜닝
# ─────────────────────────────────────────────
# pretrained=True → ImageNet으로 사전 학습된 가중치 로드
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 백본 레이어는 낮은 학습률로 미세 조정 (처음에는 freeze하고 싶다면 requires_grad=False)
backbone_params = list(model.parameters())[:-2]   # fc 레이어 제외
for p in backbone_params:
    p.requires_grad = True

# 마지막 완전 연결층(fc)을 CIFAR-10 클래스 수에 맞게 교체
in_features = model.fc.in_features   # 512
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

print(f'[INFO] 모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# ─────────────────────────────────────────────
# 3. 손실 함수 / 옵티마이저 / 스케줄러
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# 백본과 분류 헤드에 서로 다른 학습률 적용 (차등 학습률)
optimizer = optim.Adam([
    {'params': list(model.parameters())[:-2], 'lr': LR_BACKBONE},
    {'params': model.fc.parameters(),          'lr': LR},
], weight_decay=1e-4)

# 5 에폭마다 학습률을 0.1배로 감소
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ─────────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────────
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

def run_epoch(loader, training=True):
    """한 에폭 학습 또는 검증을 수행하고 (loss, accuracy)를 반환합니다."""
    model.train() if training else model.eval()
    total_loss, correct = 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct   / len(loader.dataset) * 100
    return avg_loss, accuracy


print('\n[학습 시작]')
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    va_loss, va_acc = run_epoch(val_loader,   training=False)
    scheduler.step()

    elapsed = time.time() - t0
    print(f'Epoch {epoch:02d}/{NUM_EPOCHS} | '
          f'Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}% | '
          f'Val Loss: {va_loss:.4f}  Acc: {va_acc:.2f}% | '
          f'{elapsed:.1f}s')

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(va_loss)
    history['val_acc'].append(va_acc)

# ─────────────────────────────────────────────
# 5. 학습 곡선 시각화
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs = range(1, NUM_EPOCHS + 1)
ax1.plot(epochs, history['train_loss'], label='Train')
ax1.plot(epochs, history['val_loss'],   label='Val')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

ax2.plot(epochs, history['train_acc'], label='Train')
ax2.plot(epochs, history['val_acc'],   label='Val')
ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch'); ax2.legend()

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.show()
print('[INFO] 학습 곡선 저장 완료: training_curve.png')

# ─────────────────────────────────────────────
# 6. 예측 결과 시각화 (검증 셋 첫 배치)
# ─────────────────────────────────────────────
model.eval()
images, labels = next(iter(val_loader))
with torch.no_grad():
    preds = model(images.to(DEVICE)).argmax(1).cpu()

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].permute(1, 2, 0).numpy()
    # 역정규화
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = img.clip(0, 1)
    ax.imshow(img)
    color = 'green' if preds[i] == labels[i] else 'red'
    ax.set_title(f'예측: {CLASS_NAMES[preds[i]]}\n정답: {CLASS_NAMES[labels[i]]}',
                 color=color, fontsize=7)
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
plt.show()
print('[INFO] 예측 결과 저장 완료: predictions.png')
