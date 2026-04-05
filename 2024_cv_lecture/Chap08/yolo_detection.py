"""
Chap08. 객체 탐지 (Object Detection) — YOLOv8
==============================================
Ultralytics YOLOv8을 사용해 실시간 객체 탐지를 실습합니다.

학습 내용:
  1. YOLO 모델 구조와 원리
  2. 사전 학습 모델로 즉시 추론 (Inference)
  3. 탐지 결과 시각화
  4. 커스텀 데이터셋으로 파인튜닝하는 방법 안내

실행 방법:
  python yolo_detection.py --mode image   # 이미지 탐지
  python yolo_detection.py --mode webcam  # 웹캠 실시간 탐지

요구 패키지:
  pip install ultralytics opencv-python matplotlib
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print('[ERROR] ultralytics가 설치되지 않았습니다.')
    print('        pip install ultralytics  를 실행해주세요.')
    exit(1)

# ─────────────────────────────────────────────
# 0. YOLO 모델 크기 선택
# ─────────────────────────────────────────────
# yolov8n (nano) : 가장 빠름, 정확도 낮음  → 실시간 데모에 적합
# yolov8s (small): 속도·정확도 균형       → 강의 실습 추천 ★
# yolov8m (medium): 정확도 높음
# yolov8l / yolov8x: 가장 정확, GPU 필요
MODEL_NAME = 'yolov8s.pt'   # 최초 실행 시 자동 다운로드

# COCO 80 클래스 — 탐지 가능한 객체 목록
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def draw_detections(image: np.ndarray, result) -> np.ndarray:
    """
    YOLOv8 결과(result)를 받아 바운딩 박스와 레이블을 그린 이미지를 반환합니다.
    """
    img = image.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return img  # 탐지 결과 없음

    for box in boxes:
        # 바운딩 박스 좌표 (픽셀 단위)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])          # 신뢰도 (0~1)
        cls   = int(box.cls[0])             # 클래스 인덱스
        label = f'{COCO_CLASSES[cls]} {conf:.2f}'

        # 클래스마다 색상 고정 (rainbow colormap 활용)
        color = tuple(int(c * 255) for c in plt.cm.rainbow(cls / 80)[:3])
        color = (color[2], color[1], color[0])  # RGB → BGR

        # 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 레이블 배경
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def detect_image(model, image_path: str):
    """이미지 파일에서 객체를 탐지하고 결과를 시각화합니다."""
    img = cv2.imread(image_path)
    if img is None:
        print(f'[ERROR] 이미지를 불러올 수 없습니다: {image_path}')
        return

    results = model(img, verbose=False)      # 추론 (verbose=False → 로그 숨김)
    result  = results[0]

    # 탐지된 객체 정보 출력
    print(f'\n[탐지 결과] {Path(image_path).name}')
    print(f'  총 탐지 객체 수: {len(result.boxes)}')
    for box in result.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        print(f'  - {COCO_CLASSES[cls]:<20} 신뢰도: {conf:.3f}')

    out_img = draw_detections(img, result)

    # matplotlib으로 결과 표시
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('원본 이미지', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'YOLOv8 탐지 결과 (탐지: {len(result.boxes)}개)', fontsize=13)
    axes[1].axis('off')

    plt.tight_layout()
    output_path = Path(image_path).stem + '_detected.jpg'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[INFO] 결과 저장: {output_path}')


def detect_webcam(model):
    """웹캠 영상에서 실시간 객체 탐지를 수행합니다."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[ERROR] 웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.')
        return

    print('[INFO] 웹캠 실시간 탐지 시작. 종료하려면 q 를 누르세요.')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 추론 (conf 임계값 0.4 이상만 표시)
        results = model(frame, conf=0.4, verbose=False)
        result  = results[0]
        output  = draw_detections(frame, result)

        # FPS 표시
        frame_count += 1
        fps_text = f'Objects: {len(result.boxes)}'
        cv2.putText(output, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Real-time Detection (q: quit)', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] 웹캠 탐지 종료.')


# ─────────────────────────────────────────────
# Tip: 커스텀 데이터셋 파인튜닝 방법
# ─────────────────────────────────────────────
def finetune_example():
    """
    커스텀 데이터셋으로 YOLOv8을 파인튜닝하는 방법 예시입니다.
    (실제 실행하려면 data.yaml과 이미지 데이터가 필요합니다)

    data.yaml 구조:
        path: /path/to/dataset
        train: images/train
        val: images/val
        nc: 3               # 클래스 수
        names: ['cat', 'dog', 'bird']

    YOLO 어노테이션 형식 (각 이미지마다 .txt 파일):
        <class_id> <x_center> <y_center> <width> <height>  (모두 0~1 정규화값)
    """
    model = YOLO('yolov8s.pt')
    # model.train(
    #     data='data.yaml',
    #     epochs=50,
    #     imgsz=640,
    #     batch=16,
    #     lr0=0.01,
    #     patience=10,    # 10에폭 동안 개선 없으면 조기 종료
    #     device='cuda',  # GPU 사용 시
    # )
    print('[INFO] 파인튜닝 파라미터 예시를 확인하세요 (주석 참조).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 객체 탐지 실습')
    parser.add_argument('--mode',  type=str, default='image',
                        choices=['image', 'webcam', 'finetune'],
                        help='실행 모드: image | webcam | finetune')
    parser.add_argument('--input', type=str, default='../data/bus.jpg',
                        help='입력 이미지 경로 (mode=image 일 때 사용)')
    args = parser.parse_args()

    # 모델 로드 (최초 실행 시 자동 다운로드, ~22MB)
    print(f'[INFO] YOLOv8 모델 로드 중: {MODEL_NAME}')
    model = YOLO(MODEL_NAME)

    if args.mode == 'image':
        detect_image(model, args.input)
    elif args.mode == 'webcam':
        detect_webcam(model)
    elif args.mode == 'finetune':
        finetune_example()
