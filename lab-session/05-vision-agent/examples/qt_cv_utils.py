from __future__ import annotations

from pathlib import Path

import cv2
from PyQt5.QtGui import QImage, QPixmap


def base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return base_dir() / "data"


def output_dir() -> Path:
    path = base_dir() / "outputs"
    path.mkdir(exist_ok=True)
    return path


def cv_to_qpixmap(frame) -> QPixmap:
    if frame is None:
        return QPixmap()

    if len(frame.shape) == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy())


def fit_frame(frame, max_width: int = 960, max_height: int = 720):
    if frame is None:
        return frame

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)
