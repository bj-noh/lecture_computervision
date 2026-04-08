"""
05_02_yolo_detection.py
────────────────────────────────────────────────────────────────────────────
Ultralytics YOLO 를 사용해 이미지 내 객체를 탐지합니다.

실행 예시:
    python 05_02_yolo_detection.py --image data/object.jpg

출력:
    outputs/05_02_yolo_detection.jpg
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from output_naming import with_script_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO object detection.")
    parser.add_argument("--image", type=Path, default=Path("data/object.jpg"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / with_script_prefix(__file__, "yolo_detection.jpg"),
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics: pip install ultralytics") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    result = model(str(args.image), verbose=False)[0]
    annotated = result.plot()
    Image.fromarray(annotated[..., ::-1]).save(args.output)

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        print(f"{names[cls_id]}: {conf:.4f}")


if __name__ == "__main__":
    main()
