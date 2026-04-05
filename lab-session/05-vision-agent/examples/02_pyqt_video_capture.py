from __future__ import annotations

import sys

import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from qt_cv_utils import cv_to_qpixmap, output_dir


class VideoCaptureAgent(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vision Agent - Video Capture")
        self.setGeometry(120, 120, 980, 760)

        self.cap = None
        self.frame = None
        self.captured_frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.video_label = QLabel("Press Start Camera")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)

        self.status_label = QLabel("Idle")

        start_button = QPushButton("Start Camera")
        capture_button = QPushButton("Capture Frame")
        save_button = QPushButton("Save Capture")
        stop_button = QPushButton("Stop")
        exit_button = QPushButton("Exit")

        start_button.clicked.connect(self.start_camera)
        capture_button.clicked.connect(self.capture_frame)
        save_button.clicked.connect(self.save_frame)
        stop_button.clicked.connect(self.stop_camera)
        exit_button.clicked.connect(self.close)

        button_row = QHBoxLayout()
        button_row.addWidget(start_button)
        button_row.addWidget(capture_button)
        button_row.addWidget(save_button)
        button_row.addWidget(stop_button)
        button_row.addWidget(exit_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_row)

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
        self.frame = frame
        self.video_label.setPixmap(
            cv_to_qpixmap(frame).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def capture_frame(self) -> None:
        if self.frame is None:
            self.status_label.setText("No frame available yet")
            return
        self.captured_frame = self.frame.copy()
        self.status_label.setText("Frame captured")

    def save_frame(self) -> None:
        if self.captured_frame is None:
            self.status_label.setText("Capture a frame before saving")
            return
        default_path = output_dir() / "captured_frame.jpg"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Frame", str(default_path))
        if filename:
            cv2.imwrite(filename, self.captured_frame)
            self.status_label.setText(f"Saved: {filename}")

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
    window = VideoCaptureAgent()
    window.show()
    app.exec_()
