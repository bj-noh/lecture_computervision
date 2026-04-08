"""
03_data_augmentation_demo.py
────────────────────────────────────────────────────────────────────────────
다양한 이미지 증강(Data Augmentation) 기법을 시각화합니다.

지원하는 증강 기법:
  [기본 증강]
  - RandomResizedCrop   : 랜덤 크롭 + 리사이즈
  - RandomHorizontalFlip: 좌우 반전
  - RandomRotation      : 랜덤 회전
  - ColorJitter         : 밝기/대비/채도/색조 변형
  - GaussianBlur        : 가우시안 블러
  - RandomGrayscale     : 랜덤 흑백 변환
  - RandomPerspective   : 원근 변환
  - RandomErasing       : 랜덤 영역 지우기 (Cutout)

  [고급 증강 (학습 시 사용)]
  - MixUp  : 두 이미지를 알파-블렌딩으로 섞기
  - CutMix : 한 이미지의 사각 영역을 다른 이미지로 대체

실행 예시:
    # 외부 이미지 사용
    python 03_data_augmentation_demo.py --image my_photo.jpg

    # 이미지 없으면 STL-10에서 자동 추출 (다운로드 필요 없으면 --data-root 지정)
    python 03_data_augmentation_demo.py

출력:
    outputs/aug_basic_gallery.png  ← 기본 증강 갤러리
    outputs/aug_advanced.png       ← MixUp·CutMix 결과
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from output_naming import with_script_prefix


# ── 한국어 폰트 설정 ──────────────────────────────────────────────────────
# 시스템에 설치된 NanumGothic 사용 (한글 깨짐 방지)
_KOREAN_FONT_CANDIDATES = [
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Serif CJK KR",
    "DejaVu Sans",      # 한글 없지만 fallback
]

def _set_korean_font() -> None:
    """시스템에서 한국어 지원 폰트를 찾아 matplotlib 기본 폰트로 설정."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            matplotlib.rc("font", family=name)
            matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
            print(f"[폰트] '{name}' 적용")
            return
    print("[경고] 한국어 폰트를 찾지 못했습니다. 영어로 폴백합니다.")

_set_korean_font()


# ══════════════════════════════════════════════════════════════════════════
# 샘플 이미지 로드 유틸리티
# ══════════════════════════════════════════════════════════════════════════
def load_sample_images(
    image_path: Path | None,
    data_root: Path,
    n: int = 2,
) -> list[Image.Image]:
    """
    이미지를 로드합니다.
    - image_path가 존재하면 해당 이미지를 n장 반환 (동일 이미지 복사)
    - 없으면 STL-10 / CIFAR-10 데이터셋에서 랜덤 샘플을 추출

    Args:
        image_path: 외부 이미지 경로 (없으면 None)
        data_root:  torchvision 데이터셋 루트 경로
        n:          필요한 이미지 수
    """
    if image_path is not None and image_path.exists():
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        return [img.copy() for _ in range(n)]

    print(f"[INFO] 이미지 파일 없음 → STL-10 데이터셋에서 샘플 추출")

    # STL-10이 이미 다운로드되어 있으면 사용, 아니면 CIFAR-10 시도
    try:
        dataset = datasets.STL10(root=data_root, split="test", download=False,
                                 transform=transforms.Resize((224, 224)))
        indices = random.sample(range(len(dataset)), n)
        return [TF.to_pil_image(dataset[i][0]) for i in indices]
    except Exception:
        pass

    try:
        dataset = datasets.CIFAR10(root=data_root, train=False, download=False,
                                   transform=transforms.Resize((224, 224)))
        indices = random.sample(range(len(dataset)), n)
        return [dataset[i][0] for i in indices]  # CIFAR-10은 PIL 반환
    except Exception:
        pass

    # 마지막 수단: 랜덤 노이즈 이미지 생성
    print("[WARN] 데이터셋 없음 → 랜덤 이미지 생성")
    noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img   = Image.fromarray(noise)
    return [img.copy() for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# MixUp 구현
# ══════════════════════════════════════════════════════════════════════════
def mixup(img1: Image.Image, img2: Image.Image, alpha: float = 0.4) -> tuple[Image.Image, float]:
    """
    MixUp (Zhang et al., 2018): 두 이미지를 픽셀 단위로 알파-블렌딩.

    lam ~ Beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2  (소프트 레이블)

    Args:
        img1, img2: 블렌딩할 두 PIL 이미지 (같은 크기)
        alpha:      Beta 분포 파라미터 (클수록 50:50에 가까운 혼합)

    Returns:
        (혼합된 이미지, 실제 사용된 lambda 값)
    """
    lam  = np.random.beta(alpha, alpha)  # 혼합 비율 샘플링
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    mixed = lam * arr1 + (1 - lam) * arr2           # 픽셀 단위 가중합
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed), lam


# ══════════════════════════════════════════════════════════════════════════
# CutMix 구현
# ══════════════════════════════════════════════════════════════════════════
def cutmix(img1: Image.Image, img2: Image.Image, alpha: float = 1.0) -> tuple[Image.Image, float, tuple]:
    """
    CutMix (Yun et al., 2019): img1의 사각형 영역을 img2로 대체.

    1. lam ~ Beta(alpha, alpha) 로 잘라낼 면적 비율 결정
    2. 랜덤 중심 좌표와 w, h 계산
    3. img1의 해당 영역을 img2로 복사 붙여넣기
    4. 실제 혼합 비율 lam 재계산 (잘라낸 면적 기준)

    Args:
        img1, img2: 같은 크기의 PIL 이미지
        alpha:      Beta 분포 파라미터

    Returns:
        (CutMix 이미지, 실제 lam, (x1,y1,x2,y2) 박스 좌표)
    """
    W, H = img1.size
    lam  = np.random.beta(alpha, alpha)

    # 잘라낼 박스 크기 결정
    cut_ratio  = np.sqrt(1 - lam)
    cut_w      = int(W * cut_ratio)
    cut_h      = int(H * cut_ratio)

    # 박스 중심 좌표 랜덤 샘플링
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    # img1 복사 후 해당 영역을 img2 패치로 대체
    result = img1.copy()
    patch  = img2.crop((x1, y1, x2, y2))
    result.paste(patch, (x1, y1))

    # 실제 잘라낸 면적 비율로 lam 재계산
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return result, lam, (x1, y1, x2, y2)


# ══════════════════════════════════════════════════════════════════════════
# Cutout 구현 (RandomErasing의 논문 원본 버전)
# ══════════════════════════════════════════════════════════════════════════
def cutout(img: Image.Image, n_holes: int = 1, length: int = 80) -> Image.Image:
    """
    Cutout (DeVries & Taylor, 2017): 이미지의 사각형 영역을 0으로 마스킹.

    CutMix와 달리 다른 이미지를 붙이지 않고 단순히 가림.
    → 모델이 가려진 부분을 문맥(context)으로 추론하도록 강제
    """
    result = np.array(img.copy())
    H, W   = result.shape[:2]

    for _ in range(n_holes):
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = max(cx - length // 2, 0)
        y1 = max(cy - length // 2, 0)
        x2 = min(cx + length // 2, W)
        y2 = min(cy + length // 2, H)

        result[y1:y2, x1:x2] = 0   # 해당 영역을 검정으로 마스킹

    return Image.fromarray(result)


# ══════════════════════════════════════════════════════════════════════════
# 기본 증강 갤러리 생성
# ══════════════════════════════════════════════════════════════════════════
def plot_basic_augmentations(image: Image.Image, num_samples: int, output_path: Path) -> None:
    """
    다양한 기본 torchvision 증강을 적용한 결과를 그리드로 시각화.

    각 증강 기법별 예시 1장 + 랜덤 조합(num_samples)장을 나란히 표시.
    """
    # 개별 증강 기법 목록 (이름, transform 객체)
    single_transforms = [
        ("RandomResizedCrop\n(scale=0.5~1.0)",
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0))),
        ("RandomHorizontalFlip\n(p=1.0)",
            transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomRotation\n(±30°)",
            transforms.RandomRotation(30)),
        ("ColorJitter\n(brightness/contrast/saturation)",
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
        ("GaussianBlur\n(kernel=21)",
            transforms.GaussianBlur(kernel_size=21, sigma=(2.0, 5.0))),
        ("RandomGrayscale\n(p=1.0)",
            transforms.RandomGrayscale(p=1.0)),
        ("RandomPerspective\n(distortion=0.4)",
            transforms.RandomPerspective(distortion_scale=0.4, p=1.0)),
        ("RandomErasing\n(Cutout, p=1.0)",
            transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(p=1.0, scale=(0.1, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.ToPILImage(),
            ])),
    ]

    # 랜덤 조합 증강
    random_augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    ])

    n_single = len(single_transforms)
    total    = 1 + n_single + num_samples   # 원본 + 개별 증강 + 랜덤 조합
    cols     = 4
    rows     = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes_flat = list(axes.flat)
    fig.suptitle("Data Augmentation Gallery — Basic Techniques", fontsize=14, fontweight="bold")

    # 원본 이미지
    axes_flat[0].imshow(image)
    axes_flat[0].set_title("Original", fontweight="bold", color="navy")
    axes_flat[0].axis("off")

    # 개별 증강 기법
    for i, (name, tf) in enumerate(single_transforms, start=1):
        aug_img = tf(image)
        axes_flat[i].imshow(aug_img)
        axes_flat[i].set_title(name, fontsize=8)
        axes_flat[i].axis("off")

    # 랜덤 조합 증강
    for i in range(num_samples):
        idx = 1 + n_single + i
        aug_img = random_augment(image)
        axes_flat[idx].imshow(aug_img)
        axes_flat[idx].set_title(f"Random Combo #{i+1}", fontsize=8, color="#27ae60")
        axes_flat[idx].axis("off")

    # 남는 서브플롯 숨김
    for ax in axes_flat[total:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"기본 증강 갤러리 저장: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 고급 증강 (MixUp·CutMix·Cutout) 시각화
# ══════════════════════════════════════════════════════════════════════════
def plot_advanced_augmentations(
    img1: Image.Image,
    img2: Image.Image,
    output_path: Path,
) -> None:
    """
    MixUp, CutMix, Cutout 결과를 한 장의 그림으로 비교 시각화.

    각 기법마다 lambda(혼합 비율)를 변화시키며 3가지 결과를 보여줌.
    """
    alphas      = [0.2, 0.5, 1.0]        # Beta 분포 알파값 (클수록 균등 혼합)
    cutout_lens = [40, 80, 120]          # Cutout 마스크 크기

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Advanced Data Augmentation: MixUp · CutMix · Cutout",
        fontsize=15, fontweight="bold"
    )

    # ── 행 0: 원본 이미지 두 장 ─────────────────────────────────────────
    ax = fig.add_subplot(5, 4, 1)
    ax.imshow(img1)
    ax.set_title("Image A (원본)", fontweight="bold", color="navy")
    ax.axis("off")

    ax = fig.add_subplot(5, 4, 2)
    ax.imshow(img2)
    ax.set_title("Image B (원본)", fontweight="bold", color="navy")
    ax.axis("off")

    # ── 행 1~2: MixUp ────────────────────────────────────────────────────
    for col, alpha in enumerate(alphas):
        mixed, lam = mixup(img1, img2, alpha=alpha)
        ax = fig.add_subplot(5, 4, 5 + col)
        ax.imshow(mixed)
        ax.set_title(f"MixUp\nα={alpha}, λ={lam:.2f}", fontsize=9)
        ax.axis("off")

    # MixUp 설명 텍스트
    ax_txt = fig.add_subplot(5, 4, 8)
    ax_txt.text(0.5, 0.5,
        "MixUp\n"
        "─────────────────\n"
        "x = λ·A + (1-λ)·B\n"
        "λ ~ Beta(α, α)\n\n"
        "두 이미지를 픽셀 단위로\n"
        "가중 평균하여 혼합",
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#dfe6e9", alpha=0.8),
    )
    ax_txt.axis("off")

    # ── 행 2~3: CutMix ───────────────────────────────────────────────────
    for col, alpha in enumerate(alphas):
        result, lam, (x1, y1, x2, y2) = cutmix(img1, img2, alpha=alpha)
        ax = fig.add_subplot(5, 4, 9 + col)
        ax.imshow(result)
        # 잘라낸 영역 박스 표시
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        ax.set_title(f"CutMix\nα={alpha}, λ={lam:.2f}", fontsize=9)
        ax.axis("off")

    # CutMix 설명 텍스트
    ax_txt = fig.add_subplot(5, 4, 12)
    ax_txt.text(0.5, 0.5,
        "CutMix\n"
        "─────────────────\n"
        "A의 사각 영역을\n"
        "B의 패치로 교체\n\n"
        "λ = 1 - (박스 면적)\n"
        "         /전체 면적\n\n"
        "빨간 점선: 잘라낸 영역",
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#ffeaa7", alpha=0.8),
    )
    ax_txt.axis("off")

    # ── 행 3~4: Cutout ───────────────────────────────────────────────────
    for col, length in enumerate(cutout_lens):
        result = cutout(img1, n_holes=1, length=length)
        ax = fig.add_subplot(5, 4, 13 + col)
        ax.imshow(result)
        ax.set_title(f"Cutout\nlength={length}", fontsize=9)
        ax.axis("off")

    # Cutout 설명 텍스트
    ax_txt = fig.add_subplot(5, 4, 16)
    ax_txt.text(0.5, 0.5,
        "Cutout\n"
        "─────────────────\n"
        "이미지의 사각 영역을\n"
        "검정(0)으로 마스킹\n\n"
        "다른 이미지 불필요\n"
        "단일 이미지 증강",
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#b2bec3", alpha=0.8),
    )
    ax_txt.axis("off")

    # ── 행 4: MixUp vs CutMix 비교 (λ=0.5 고정) ─────────────────────────
    mixed_fixed, lam_mix   = mixup(img1, img2, alpha=1.0)  # alpha=1 → 균등
    cut_fixed, lam_cut, _  = cutmix(img1, img2, alpha=1.0)

    ax = fig.add_subplot(5, 4, 17)
    ax.imshow(img1)
    ax.set_title("Image A", fontsize=9, color="navy")
    ax.axis("off")

    ax = fig.add_subplot(5, 4, 18)
    ax.imshow(img2)
    ax.set_title("Image B", fontsize=9, color="navy")
    ax.axis("off")

    ax = fig.add_subplot(5, 4, 19)
    ax.imshow(mixed_fixed)
    ax.set_title(f"MixUp (λ={lam_mix:.2f})\n[두 이미지가 겹쳐 보임]", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(5, 4, 20)
    ax.imshow(cut_fixed)
    ax.set_title(f"CutMix (λ={lam_cut:.2f})\n[영역이 잘려 붙여짐]", fontsize=8)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"고급 증강 비교 그림 저장: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Data augmentation visualization demo.")
    parser.add_argument("--image",       type=Path, default=None,
                        help="시각화에 사용할 이미지 경로 (없으면 데이터셋에서 자동 추출)")
    parser.add_argument("--num-samples", type=int,  default=6,
                        help="랜덤 조합 증강 샘플 수")
    parser.add_argument("--data-root",   type=Path, default=Path("data"),
                        help="torchvision 데이터셋 루트")
    parser.add_argument("--output-dir",  type=Path, default=Path("outputs"))
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    # 재현성을 위한 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 샘플 이미지 로드 ─────────────────────────────────────────────────
    imgs = load_sample_images(args.image, args.data_root, n=2)
    img1, img2 = imgs[0], imgs[1]

    print(f"이미지 크기: {img1.size}")

    # ── 기본 증강 갤러리 ─────────────────────────────────────────────────
    plot_basic_augmentations(
        image       = img1,
        num_samples = args.num_samples,
        output_path = args.output_dir / with_script_prefix(__file__, "aug_basic_gallery.png"),
    )

    # ── 고급 증강 (MixUp·CutMix·Cutout) ─────────────────────────────────
    plot_advanced_augmentations(
        img1        = img1,
        img2        = img2,
        output_path = args.output_dir / with_script_prefix(__file__, "aug_advanced.png"),
    )

    print("\n완료! 저장된 파일:")
    print(f"  {args.output_dir / with_script_prefix(__file__, 'aug_basic_gallery.png')}")
    print(f"  {args.output_dir / with_script_prefix(__file__, 'aug_advanced.png')}")


if __name__ == "__main__":
    main()
