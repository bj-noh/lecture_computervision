<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<h1 align="center">🎓 컴퓨터 비전 강의</h1>
<p align="center"><b>Computer Vision Lecture — 대학교 3·4학년 대상</b></p>
<p align="center">
  고전 영상처리의 수학적 원리부터 Vision Transformer, Diffusion Model까지<br>
  <b>단계적으로</b> 배우는 컴퓨터 비전 강의 자료입니다.
</p>

---

## 📚 강의 목차

### 🔷 Part 1 · OpenCV 기초 (Classical CV)

| 챕터 | 주제 | 핵심 개념 |
|------|------|-----------|
| Chap 01 | OpenCV 입문 | 이미지 읽기/저장, 웹캠, 드로잉, 마우스 이벤트 |
| Chap 02 | 색 공간과 히스토그램 | BGR↔HSV, Back-Projection, 감마 보정 |
| Chap 03 | 필터링과 모폴로지 | 가우시안·LoG 필터, Dissolve 효과 |
| Chap 04 | 엣지와 윤곽선 | Sobel, Canny, HoughLines/Circles, Contour |
| Chap 05 | 특징점과 기술자 | SIFT, FLANN 매칭, 호모그래피 |
| Chap 06 | 객체 추적 | KNN, SVM+HOG, MeanShift, CAMShift, Optical Flow |

### 🔶 Part 2 · 딥러닝 기반 컴퓨터 비전 (Deep Learning CV)

| 챕터 | 주제 | 핵심 개념 |
|------|------|-----------|
| Chap 07 | CNN & 전이학습 | ResNet18 fine-tuning, CIFAR-10 분류 |
| Chap 08 | 객체 탐지 | YOLOv8, Faster R-CNN, mAP 평가 |
| Chap 09 | 시맨틱 분할 | U-Net from scratch, IoU 평가 |
| Chap 10 | Vision Transformer | ViT 구조 이해, fine-tuning |
| Chap 11 | 생성 모델 · 확산 모델 | DCGAN → DDPM 흐름 이해 |
| Chap 13 | GAN 심화 | DCGAN, Fashion-MNIST 이미지 생성 |

---

## 🗂 디렉토리 구조

```
lecture_computervision/
├── 2023_cv_lecture/
│   ├── data/          # 실습 이미지·영상
│   ├── lecture/       # PDF 강의자료
│   └── src/           # 챕터별 Python 실습 코드
│
└── 2024_cv_lecture/
    ├── data/          # 실습 이미지·영상 (CIFAR-10 포함)
    ├── Chap01/        # OpenCV 기초
    ├── Chap03/        # 필터·효과
    ├── Chap05/        # SIFT + FLANN
    ├── Chap07/        # CNN & 전이학습  ← NEW
    ├── Chap08/        # 객체 탐지       ← NEW
    ├── Chap09/        # 시맨틱 분할     ← NEW
    ├── Chap10/        # Vision Transformer ← NEW
    ├── Chap11/        # 확산 모델       ← NEW
    └── Chap13/        # GAN
```

---

## ⚙️ 개발 환경 설정

### 1. Python 가상 환경 생성

```bash
python -m venv cv_env
source cv_env/bin/activate      # macOS / Linux
# cv_env\Scripts\activate       # Windows
```

### 2. 패키지 설치

```bash
# OpenCV + 기본 과학 계산
pip install opencv-python numpy matplotlib scikit-image

# 딥러닝 (PyTorch — GPU 버전은 https://pytorch.org 참고)
pip install torch torchvision torchaudio

# 최신 CV 라이브러리
pip install ultralytics          # YOLOv8
pip install timm                 # Vision Transformer 모델 허브
pip install albumentations       # 고급 데이터 증강
```

### 3. 실행 확인

```python
import cv2, torch, timm
print('OpenCV:', cv2.__version__)
print('PyTorch:', torch.__version__)
print('CUDA 사용 가능:', torch.cuda.is_available())
```

---

## 🚀 챕터별 빠른 시작

```bash
# Chap01 — 이미지 읽기
python 2024_cv_lecture/Chap01/P01_image_read.py

# Chap07 — ResNet18 전이학습 (CIFAR-10)
python 2024_cv_lecture/Chap07/transfer_learning.py

# Chap08 — YOLOv8 객체 탐지
python 2024_cv_lecture/Chap08/yolo_detection.py

# Chap09 — U-Net 시맨틱 분할
python 2024_cv_lecture/Chap09/unet_segmentation.py

# Chap10 — Vision Transformer 분류
python 2024_cv_lecture/Chap10/vit_classification.py

# Chap11 — DDPM 확산 모델
python 2024_cv_lecture/Chap11/ddpm_diffusion.py
```

---

## 🗺 학습 로드맵

```
[이미지 기초]──►[필터·엣지]──►[특징점 매칭]
                                    │
                                    ▼
                        [CNN + 전이학습] (Chap07)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             [객체 탐지]    [시맨틱 분할]   [Vision Transformer]
             (Chap08)       (Chap09)         (Chap10)
                    │
                    ▼
             [생성 모델 · 확산 모델]
             (Chap11, Chap13)
```

---

## 📌 참고 자료

| 자료 | 링크 |
|------|------|
| OpenCV 공식 문서 | https://docs.opencv.org |
| PyTorch 튜토리얼 | https://pytorch.org/tutorials |
| Papers With Code | https://paperswithcode.com |
| Hugging Face CV | https://huggingface.co/docs/transformers/tasks/image_classification |

---

## 📝 라이선스

강의 자료는 교육 목적으로 자유롭게 활용할 수 있습니다.
