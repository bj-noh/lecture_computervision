from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from qt_cv_utils import cv_to_qpixmap


class ColorZoneAlertAgent(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vision Agent - Color Zone Alert")
        self.setGeometry(120, 120, 980, 760)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.video_label = QLabel("Press Start Camera")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.status_label = QLabel("Watching for strong red/orange/yellow regions in the center ROI.")

        start_button = QPushButton("Start")
        stop_button = QPushButton("Stop")
        exit_button = QPushButton("Exit")

        start_button.clicked.connect(self.start_camera)
        stop_button.clicked.connect(self.stop_camera)
        exit_button.clicked.connect(self.close)

        row = QHBoxLayout()
        row.addWidget(start_button)
        row.addWidget(stop_button)
        row.addWidget(exit_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_camera(self) -> None:
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Could not open webcam")
            return
        self.timer.start(30)
        self.status_label.setText("Camera started")

    def update_frame(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_label.setText("Failed to read frame")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red_a = np.array([0, 100, 100], dtype=np.uint8)
        upper_red_a = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_b = np.array([160, 100, 100], dtype=np.uint8)
        upper_red_b = np.array([179, 255, 255], dtype=np.uint8)
        lower_yellow = np.array([15, 90, 90], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_red_a, upper_red_a)
        mask |= cv2.inRange(hsv, lower_red_b, upper_red_b)
        mask |= cv2.inRange(hsv, lower_yellow, upper_yellow)

        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.25), int(h * 0.2)
        x2, y2 = int(w * 0.75), int(h * 0.8)
        roi_mask = mask[y1:y2, x1:x2]
        ratio = float(np.count_nonzero(roi_mask)) / roi_mask.size

        alert = ratio > 0.12
        color = (0, 0, 255) if alert else (0, 255, 0)
        text = f"Warning zone ratio: {ratio:.2%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        if alert:
            self.status_label.setText("Alert: strong warning-zone color detected in ROI")
            QApplication.beep()
        else:
            self.status_label.setText("Safe: monitored region is below threshold")

        self.video_label.setPixmap(
            cv_to_qpixmap(frame).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_label.setText("Camera stopped")

    def closeEvent(self, event) -> None:
        self.stop_camera()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorZoneAlertAgent()
    window.show()
    app.exec_()
