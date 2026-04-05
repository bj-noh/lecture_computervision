from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO on a sample image.")
    parser.add_argument("--image", type=Path, default=Path("data/view.jpg"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_result.jpg"))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics: pip install ultralytics") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    result = model(str(args.image), verbose=False)[0]
    annotated = result.plot()
    cv2.imwrite(str(args.output), annotated)

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        print(f"{names[cls_id]}: {conf:.4f}")


if __name__ == "__main__":
    main()
